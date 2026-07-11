"""FML Floor → Luau LFR emitter.

Single public function: `emit_lua(floor, source_path, version) -> str`.

See `docs/design/LFR.md` for the canonical output format.

Prose template compilation:
  ``_compile_prose(prose_val, prop_key, source_path)`` lowers a ``ProseValue``
  to a Luau ``function(self, ctx) ... end`` closure per ``docs/design/PROSE.md``.

Kind-property inheritance:
  Before emitting an entity's property table, ``_resolve_inherited_properties``
  walks the kind chain (instance → parent kind → grandparent → … → root) and
  merges properties bottom-up so root defaults appear first and instance values
  win.  Triggers follow the same rule: instance trigger for a stage name
  overrides the kind's trigger of the same name; otherwise the kind's trigger
  is used.  See ``docs/design/OBJECT_MODEL.md`` §7 (Parent chain).

Tree-shake pass:
  ``emit_lua`` accepts an optional ``kept`` set of entity ids produced by
  ``parser.tree_shake.tree_shake()``.  When supplied, only entities whose id
  appears in ``kept`` are emitted.  Verbs and kind-chain ancestors are always
  included by the tree-shake pass (see ``docs/design/PARSER.md`` §9).
  When ``kept`` is ``None`` (the default) all entities are emitted — the
  pre-tree-shake conservative behaviour.
"""

from __future__ import annotations

import re
from typing import Any

from .dice_value import LuauCode, ProseValue
from .errors import FmlSyntaxError
from .models import ActionLine, BareLink, FMLEntity, Floor, OutputLine, Predicate, PropertySet, Trigger, TriggerBodyItem, UnderstandDirective

# ─── Section grouping ─────────────────────────────────────────────────────────

# Kinds that carry stdlib metadata, not floor-data — skip them entirely.
_SKIP_KINDS = frozenset(["kind_definition", "verb"])

# Lua identifier pattern — keys that don't match get bracket syntax.
_LUA_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# Trigger stage word → LFR prefix, including the InsteadOf special case.
_STAGE_MAP: dict[str, str] = {
    "Test":       "test",
    "InsteadOf":  "instead_of",
    "Before":     "before",
    "On":         "on",
    "After":      "after",
    "Report":     "report",
}


# ─── Kind-property inheritance ────────────────────────────────────────────────


def _resolve_inherited_properties(
    entity: FMLEntity, floor: Floor
) -> tuple[dict[str, Any], list[Trigger]]:
    """Walk the entity's kind chain and return fully-merged (properties, triggers).

    Algorithm (OBJECT_MODEL.md §7 — parent chain only, no applique/mix-in):

    1. Build the chain [instance_kind, parent_kind, grandparent_kind, …].
    2. For each kind in the chain (root first, instance last), look up the
       ``kind_definition`` entity in ``floor.entities`` by its ``name`` property.
    3. Merge properties root→…→parent→instance, instance wins per key.
       - Scalar / function / prose: instance slot replaces parent slot.
       - List: instance list replaces parent list (no concatenation — per spec).
    4. Merge triggers by stage-slot key (``_trigger_slot_key(trigger.name)``):
       instance trigger replaces parent trigger of the same slot.

    Returns the merged (properties_dict, triggers_list) ready to emit.
    The entity's own ``kind`` and ``id`` / ``name`` fields are NOT in properties.
    Skipped properties: ``name``, ``ancestor`` — these are kind-definition
    metadata, not inheritable world properties.
    """
    # Build kind entity lookup: kind name → kind_definition FMLEntity
    kind_defs: dict[str, FMLEntity] = {}
    for ent in floor.entities.values():
        if ent.kind == "kind_definition":
            kname = ent.properties.get("name")
            if isinstance(kname, str):
                kind_defs[kname] = ent

    # Walk the chain from root to leaf so deeper (more-specific) kinds win.
    chain = floor.kind_chain(entity.kind)  # [instance_kind, parent, …, root]
    # Reverse so we merge root → … → parent → instance (instance wins).
    chain_reversed = list(reversed(chain))

    # Skip internal kind-definition metadata keys — not world-entity properties.
    # 'kind' appears in properties because the parser stores it there as well as
    # on entity.kind; we don't want 'kind: kind_definition' bleeding into instances.
    _KD_SKIP = frozenset({"name", "ancestor", "attributes", "kind"})

    merged_props: dict[str, Any] = {}
    # Triggers keyed by slot (e.g. "after:Attack") so instance overrides parent.
    trigger_by_slot: dict[str, Trigger] = {}

    for kind_name in chain_reversed:
        kd = kind_defs.get(kind_name)
        if kd is None:
            continue
        # Merge kind's properties (excluding internal metadata).
        for k, v in kd.properties.items():
            if k not in _KD_SKIP:
                merged_props[k] = v
        # Merge kind's triggers (slot-keyed, later overrides earlier).
        for trigger in kd.triggers:
            slot = _trigger_slot_key(trigger.name)
            trigger_by_slot[slot] = trigger

    # Instance properties win over all kind-chain properties.
    for k, v in entity.properties.items():
        merged_props[k] = v

    # Instance triggers win over kind-chain triggers of the same slot.
    for trigger in entity.triggers:
        slot = _trigger_slot_key(trigger.name)
        trigger_by_slot[slot] = trigger

    # Rebuild trigger list in a stable order: kind-chain triggers first (by
    # their slot), then instance-only triggers appended. Order within each
    # group is preserved from the source.
    seen_slots: set[str] = set()
    merged_triggers: list[Trigger] = []

    # Kind-chain triggers (in reverse-chain order, then slot order)
    for kind_name in chain_reversed:
        kd = kind_defs.get(kind_name)
        if kd is None:
            continue
        for trigger in kd.triggers:
            slot = _trigger_slot_key(trigger.name)
            if slot not in seen_slots:
                # Only add if this is the winning trigger for the slot.
                if trigger_by_slot[slot] is trigger:
                    merged_triggers.append(trigger)
                    seen_slots.add(slot)

    # Instance triggers: add all (already de-dup'd by slot if overriding kind).
    for trigger in entity.triggers:
        slot = _trigger_slot_key(trigger.name)
        if slot not in seen_slots:
            merged_triggers.append(trigger)
            seen_slots.add(slot)

    return merged_props, merged_triggers


# ─── Stage name → table key ───────────────────────────────────────────────────

def _verb_stage_key(trigger_name: str) -> str:
    """'Test Go' → 'test', 'InsteadOf Go' → 'instead_of', 'On Take' → 'on'."""
    parts = trigger_name.split(None, 1)
    if not parts:
        return "on"
    return _STAGE_MAP.get(parts[0], parts[0].lower())


# Stage words that prefix a trigger heading in the legacy <stage> <Event> form.
_TRIGGER_STAGE_WORDS = frozenset(
    ["test", "insteadof", "instead_of", "before", "on", "after", "report"]
)


def _om_event_name(trigger_name: str) -> str:
    """Map an FML trigger heading to the om event name (SCENES_AND_ACTS §3).

    The object-model event bus has a single post-commit event per occurrence
    (``OnEnter``/``OnOpen``/…); the legacy ``<stage> <Event>`` distinction
    collapses to ``On<Event>``. Examples:
      'On Open'          → 'OnOpen'
      'After Enter'      → 'OnEnter'
      'On RoundStart'    → 'OnRoundStart'
      'InsteadOf Damaged'→ 'OnDamaged'   (veto semantics are O4, not yet honoured)
    The leading stage word is dropped; remaining words are concatenated with the
    first letter of each upper-cased, prefixed with 'On'.
    """
    parts = trigger_name.split()
    # empty, or a bare stage word with no event component → just "On"
    if not parts or (len(parts) == 1 and parts[0].lower() in _TRIGGER_STAGE_WORDS):
        return "On"
    if len(parts) > 1 and parts[0].lower() in _TRIGGER_STAGE_WORDS:
        event_words = parts[1:]
    else:
        event_words = parts
    return "On" + "".join(w[:1].upper() + w[1:] for w in event_words)


# Interception stages — triggers that BLOCK/redirect an action rather than react
# to it. In the object model they lower to a Before-stage SELF behaviour on the
# reach-path (OBJECT_MODEL §5.6 / SCENES_AND_ACTS O4), able to ctx:veto. The
# post-commit On/After/Report stages lower to om.on reactions instead.
_OM_INTERCEPTION_STAGES = frozenset(["test", "instead_of", "before"])


def _om_interception_verb(trigger_name: str) -> str:
    """The verb an interception trigger guards: the event word(s) after the
    stage, lowercased and hyphen-joined. 'Test Take'→'take', 'InsteadOf Attack'
    →'attack', 'Before Put In'→'put-in'."""
    parts = trigger_name.split()
    if len(parts) > 1 and parts[0].lower() in _TRIGGER_STAGE_WORDS:
        event_words = parts[1:]
    else:
        event_words = parts
    return "-".join(w.lower() for w in event_words)


# ─── Graph emitter (F6 — binding-surface LFR) ────────────────────────────────

# Built-in verbs the engine bootstraps (tg_verb_bootstrap): take, drop, put-in, go.
# Re-emitting any of these via define_verb fails on the duplicate name, so skip
# them. The engine names "put-in" with a hyphen, but FML EntityIds are snake_case
# (^[a-z][a-z0-9_]*$), so a content verb intended as put-in is slugged "put_in" —
# match that form. ("put" alone is NOT a built-in and must remain emittable.)
_BUILTIN_VERBS = frozenset(["take", "drop", "put_in", "go"])

# Kinds that are schema metadata, not world instances.
_SCHEMA_KINDS = frozenset(["kind_definition", "verb", "event"])


def emit_lua_om(
    floor: Floor,
    source_path: str | None = None,
    stdlib_module: bool = False,
) -> str:
    """Emit an object-model (prototype) LFR for the ``--engine om`` dispatch path.

    Thin wrapper over :func:`emit_lua_graph` with ``om=True``. The output is the
    same DB-init shape but:

    * every world entity gets a **prototype edge** to its kind node — resolved
      at load time from the ``kind:<name>`` registry on ``object`` (set by the
      Lua trampoline + any om stdlib module); unknown kinds fall through to
      ``object``. The string ``kind`` property is dropped (the edge replaces it).
    * verbs are emitted **grammar-only** (``engine.define_verb`` with no stages)
      so the om parser can resolve nouns; their behaviour is Model-A fragments
      (``om.set_behaviour``) — verb-stage behaviour is a later phase.
    * instance lua/luau **reaction triggers** lower to Layer-B subscriptions
      (SCENES_AND_ACTS §3): ``om.on(node, "On<Event>", function(ctx) … end)``,
      fired by the engine/trampoline event bus. (Bodies authored against the
      legacy host API — ``ctx.noun``, ``engine.set_property``/``set_world`` — do
      not resolve against the om reaction ctx + graph store and need a content
      port; the lowering itself is correct.)

    Scope: structural lowering + Layer-B reaction subscriptions. Verb-stage
    behaviour (om.set_behaviour with roles) and FML action-vocabulary bodies
    (Set/Trigger/Block) are later phases.
    """
    return emit_lua_graph(
        floor, source_path=source_path, stdlib_module=stdlib_module, om=True
    )


def emit_lua_graph(
    floor: Floor,
    source_path: str | None = None,
    stdlib_module: bool = False,
    om: bool = False,
) -> str:
    """Emit a DB-init LFR for the LPG engine binding surface.

    Calls ``engine.*`` bindings to populate the graph and register verbs.

    When ``om=True`` the output targets the prototype object-model dispatch
    (``--engine om``): see :func:`emit_lua_om`. The ``om`` branches are additive
    and guarded so ``om=False`` is byte-for-byte the legacy graph output.

    Two modes (mirroring the legacy emit_lua / emit_lua_stdlib_module split):

    * **stdlib module** (``stdlib_module=True``): emit SCHEMA + PROCEDURES only —
      ``define_kind`` + ``define_relation`` + ``define_verb``. No world instances,
      no start actor. This is loaded first in the assembly (``--stdlib``).
    * **floor** (default): emit INSTANCES — ``create_node`` + ``relate`` +
      floor-local ``define_verb`` + ``set_start_actor``. Entities/verbs that came
      from a stdlib import (``floor.stdlib_entity_ids``) are skipped — the stdlib
      module already registered them; the floor only owns its own content.

    Emission order: header → [stdlib: kinds] → relations(map) → create_node →
    relate → define_verb (skip engine built-ins) → [floor: set_start_actor].

    Deterministic: same input → bit-identical output.
    """
    src = source_path or "unknown.md"
    parts: list[str] = []

    # Scope filter: in floor mode, a non-None stdlib_entity_ids means "only emit
    # entities this floor owns" (skip imported stdlib catalog/kinds/verbs). In
    # stdlib-module mode every entity is in scope.
    stdlib_ids = floor.stdlib_entity_ids or set()

    def _in_scope(ent: FMLEntity) -> bool:
        if stdlib_module:
            return True
        return ent.id not in stdlib_ids

    # 1. Header ----------------------------------------------------------------
    lua_path = src.replace(".md", ".lua") if src else "unknown.lua"
    parts.append(f"-- {lua_path}")
    emitter_label = "object-model emitter" if om else "graph emitter"
    parts.append(f"-- Generated by fml-parser ({emitter_label}) from {src}")
    hash_val = floor.fml_source_hash or "unknown"
    parts.append(f"-- fml-source-hash: {hash_val}")
    parts.append("")

    # om floor: a helper to set an instance's prototype from its declared name,
    # resolved at load time via the wyrd.named name→node registry (the trampoline
    # + any om stdlib module register the base prototypes there). Unknown names
    # fall through to object's defaults — no set_prototype, no error. ("kind" has
    # dissolved into "a named prototype"; the registry is wyrd.named, not kind:.)
    if om and not stdlib_module:
        parts.append("local function _proto(n, name)")
        parts.append("    -- name: a prototype name (e.g. \"room\"); resolve the node")
        parts.append("    -- registered under that name and set it as n's prototype.")
        parts.append('    local k = wyrd.named(name)')
        parts.append("    if k then wyrd.set_prototype(n, k) end")
        parts.append("end")
        parts.append("")
    elif om and stdlib_module:
        # P6a: the stdlib+om path is structurally lowered but does NOT yet
        # register kinds into the kind:<name> registry the floor's _proto helper
        # reads (that needs the full om kind-node emission, deferred to the
        # behaviour-port phase). A floor lowered with --om resolves its core
        # kinds from the engine trampoline; until the stdlib is ported, custom
        # stdlib kinds fall through to object. Flag it so this isn't shipped as
        # a complete om stdlib by mistake.
        parts.append(
            "-- WARNING: --stdlib-module --om is P6a-incomplete: kinds are NOT yet"
        )
        parts.append(
            "-- registered in the om kind:<name> registry (behaviour-port phase)."
        )
        parts.append("")

    # 2. Kinds -----------------------------------------------------------------
    # Only the stdlib module registers kinds; a floor relies on the stdlib it
    # imports (loaded first in the assembly) to have defined them.
    # Emit in section_mappings order when available, else stable declaration order.
    kind_entities: list[FMLEntity] = []
    if not stdlib_module:
        kind_entities = []
    elif floor.section_mappings:
        # Collect kind names from section_mappings (preserves declared order).
        seen_kind_names: set[str] = set()
        kind_name_order: list[str] = []
        for _heading, kind_id in floor.section_mappings:
            if kind_id not in seen_kind_names:
                kind_name_order.append(kind_id)
                seen_kind_names.add(kind_id)
        # Build a lookup: kind-definition entity by its `name` property.
        kdef_by_name: dict[str, FMLEntity] = {}
        for ent in floor.entities.values():
            if ent.kind == "kind_definition":
                kname = ent.properties.get("name")
                if isinstance(kname, str):
                    kdef_by_name[kname] = ent
        # Emit in section-mapping order, then any remaining kind_definition entities.
        seen_ids: set[str] = set()
        for kname in kind_name_order:
            kd = kdef_by_name.get(kname)
            if kd is not None and kd.id not in seen_ids:
                kind_entities.append(kd)
                seen_ids.add(kd.id)
        for ent in floor.entities.values():
            if ent.kind == "kind_definition" and ent.id not in seen_ids:
                kind_entities.append(ent)
                seen_ids.add(ent.id)
    else:
        kind_entities = [e for e in floor.entities.values() if e.kind == "kind_definition"]

    if kind_entities:
        parts.append("-- Kinds")
    for ent in kind_entities:
        kname = ent.properties.get("name")
        if not isinstance(kname, str):
            kname = ent.name
        parts.append(f"engine.define_kind({_lua_string(kname)})")
    if kind_entities:
        parts.append("")

    # 3. Custom relations — emit 'map' if any world entity declares exits -------
    # World instances are a floor concern; in stdlib-module mode there are none.
    world_entities = (
        []
        if stdlib_module
        else [
            e for e in floor.entities.values()
            if e.kind not in _SCHEMA_KINDS and _in_scope(e)
        ]
    )
    need_map_relation = any(
        e.properties.get("exits") or e.properties.get("exit")
        for e in world_entities
    )
    if need_map_relation:
        parts.append("-- Custom relations")
        parts.append(
            'local _rel_map = engine.define_relation("map", { symmetric = true, role = "map" })'
        )
        parts.append("")

    # 4. World entity nodes ---------------------------------------------------
    if world_entities:
        parts.append("-- World entities")
    for ent in world_entities:
        scalar_props = _collect_scalar_props(ent)
        # om: the `kind` is now a prototype EDGE (emitted below), not a string
        # property — drop it from the node's own bag.
        if om:
            scalar_props = {k: v for k, v in scalar_props.items() if k != "kind"}
        pair_strs = [
            f"{_lua_key(k)} = {_lua_value(v)}" for k, v in scalar_props.items()
        ]
        # Noun aliases: a list value (so _collect_scalar_props drops it) lowered
        # to a single pipe-delimited lowercased string ("|cup|bone cup|") that the
        # play-path's noun resolver matches, so "take cup" resolves the Bone Cup.
        alias_blob = _alias_blob(ent)
        if alias_blob is not None:
            pair_strs.append(f"aliases = {_lua_string(alias_blob)}")
        if pair_strs:
            parts.append(
                f"local n_{ent.id} = engine.create_node({{ name = {_lua_string(ent.name)}, "
                f"{', '.join(pair_strs)} }})"
            )
        else:
            parts.append(
                f"local n_{ent.id} = engine.create_node({{ name = {_lua_string(ent.name)} }})"
            )
        # om: set the prototype edge from the entity's kind (resolved by name).
        if om and isinstance(ent.kind, str) and ent.kind:
            parts.append(f"_proto(n_{ent.id}, {_lua_string(ent.kind)})")
    if world_entities:
        parts.append("")

    # 4a. Name registry — register each world entity under its display name so
    # runtime content can resolve it with `wyrd.named("<Name>")`, e.g. cloning a
    # monster prototype at ambush time:
    #     wyrd.create(wyrd.named("Skeleton Scavenger"))
    # Mirrors the trampoline registering its own prototypes (OBJECT_MODEL §3 — a
    # "kind" is a named prototype). Without this, only stdlib kinds are
    # name-addressable and a floor's own entities are invisible to wyrd.named,
    # so a runtime `wyrd.create(wyrd.named(...))` clones a stat-less bare node.
    # Deterministic order; on a duplicate display name the last entity wins.
    # NB: wyrd.named is ONE flat namespace shared with kinds/prototypes (see the
    # _proto helper above, which resolves kind names through it), so a floor
    # entity whose display name equals a kind name (e.g. "npc"/"room") shadows
    # that prototype for runtime lookups. Runs after the _proto loop, so
    # load-time prototype resolution is unaffected; only later wyrd.named calls
    # see the entity. Authors keep entity display names distinct from kind names.
    if om and world_entities:
        parts.append("-- Name registry (wyrd.named lookups)")
        for ent in world_entities:
            parts.append(f"wyrd.register({_lua_string(ent.name)}, n_{ent.id})")
        parts.append("")

    # 4b. Prose — an entity's prose lowers to a function(self, ctx) (PROSE.md),
    # registered with engine.set_prose so the engine renders it at look/examine
    # time (evaluating any inline conditionals). Markdown links are flattened.
    prose_lines: list[str] = []
    for ent in world_entities:
        fn = _prose_function_literal(ent)
        if fn is not None:
            prose_lines.append(f"engine.set_prose(n_{ent.id}, {fn})")
    if prose_lines:
        parts.append("-- Prose (set_prose: function(self, ctx))")
        parts.extend(prose_lines)
        parts.append("")

    # 4c. Persona `goals` list — a list property authored in a ``#### Persona``
    # H4 subsection.  These cannot be inlined into create_node (which takes
    # only scalars), so they emit as engine.set_prop with a Lua table.  The
    # engine (wyrd #135) supports TG_VALUE_LIST on node properties; the emitter
    # pre-wires it here so floors authoring #### Persona are ready when the
    # wyrd binding surface lands.  Only `goals` is treated this way — other
    # list-typed content properties (e.g. `connects`, `aliases`) have their own
    # dedicated emission paths and must not be re-emitted here.
    goals_lines: list[str] = []
    for ent in world_entities:
        goals_val = ent.properties.get("goals")
        if isinstance(goals_val, list) and goals_val:
            if all(isinstance(item, str) for item in goals_val):
                items_lua = ", ".join(_lua_string(item) for item in goals_val)
                goals_lines.append(
                    f'engine.set_prop(n_{ent.id}, "goals", {{ {items_lua} }})'
                )
    if goals_lines:
        parts.append("-- NPC goals (set_prop with list value; wyrd #135)")
        parts.extend(goals_lines)
        parts.append("")

    # 5. Relations (after all nodes exist) ------------------------------------
    world_ids: set[str] = {e.id for e in world_entities}
    relation_lines: list[str] = []

    for ent in world_entities:
        # Containment: at_location / location property → relate("in", ...).
        # (FML uses `at_location` for items in a room; `location` is the older
        # form. First match wins.)
        # §22 Phase 5: an optional `position: [x, y, z]` rides the location link
        # as the engine's integer cell payload (#104). Validate it up-front so a
        # malformed position is always reported — even on an entity with no
        # recognized container (where the position requires `location`/
        # `at_location` and is otherwise a no-op).
        cell = _parse_cell(ent.properties.get("position"), ent.id)
        container = ent.properties.get("at_location") or ent.properties.get("location")
        if isinstance(container, str) and container in world_ids:
            if cell is not None:
                x, y, z = cell
                relation_lines.append(
                    f'engine.relate("in", n_{ent.id}, n_{container}, {x}, {y}, {z})'
                )
            else:
                relation_lines.append(
                    f'engine.relate("in", n_{ent.id}, n_{container})'
                )

    # Room entrances (#119): the cell an arriving actor lands at, so a spawned
    # player appears at an authored spot rather than the container origin (where
    # an unpositioned NPC/boss rests). An authored `entrance: [x,y,z]` wins; the
    # south-centre fallback for mapped rooms is injected by strip_map_keys (which
    # still has the map dims), so here we just emit whatever `entrance` resolved.
    entrance_lines: list[str] = []
    for ent in world_entities:
        entrance = _parse_cell(ent.properties.get("entrance"), ent.id, "entrance")
        if entrance is not None:
            ex, ey, ez = entrance
            entrance_lines.append(f"engine.set_entrance(n_{ent.id}, {ex}, {ey}, {ez})")

    # Exits → 'map' edges. The relation is symmetric, so emit each unordered
    # room pair once and skip self-loops (avoids duplicate/degenerate edges from
    # rooms that declare reciprocal or self exits).
    seen_map_pairs: set[frozenset[str]] = set()
    exit_prop_lines: list[str] = []
    for ent in world_entities:
        exits = ent.properties.get("exits") or ent.properties.get("exit")
        if not isinstance(exits, dict):
            continue
        for direction, dest in exits.items():
            dest_slug = _exit_dest_slug(dest)
            if not (isinstance(dest_slug, str) and dest_slug in world_ids):
                continue
            # Directional exit: a per-room property the engine's `go` reads, so
            # "go north" / "go n" / bare "north" resolve to the destination.
            # Directions are DATA-DRIVEN: a known alias folds to its canonical
            # form (n→north); any other author-defined direction (forward /
            # clockwise / rock / paper / scissors) passes through as exit_<name>.
            # The engine resolves the standard set via its registry and falls
            # back to a bare exit_<word> for the custom ones.
            raw = str(direction).strip().lower()
            canon = _DIR_CANON.get(raw, raw)
            if _DIR_KEY_RE.fullmatch(canon):
                exit_prop_lines.append(
                    f'engine.set_prop(n_{ent.id}, "exit_{canon}", n_{dest_slug})'
                )
                # A `{room, door}` exit also records the gating door per
                # direction (exit_door_<canon>), which the engine surfaces to the
                # go verb. Declared on each side's exit → two-way gating.
                door_slug = _exit_door_slug(dest)
                if (door_slug is not None and door_slug in world_ids
                        and _DIR_KEY_RE.fullmatch(canon)):
                    exit_prop_lines.append(
                        f'engine.set_prop(n_{ent.id}, "exit_door_{canon}", n_{door_slug})'
                    )
                # Per-exit entry override (#119): leaving this room via `canon`
                # lands the mover at an authored cell in the destination (the
                # object-form exit `<dir>: {room, enter_at: [x,y,z]}`), overriding
                # the destination's default entrance.
                enter_cell = _parse_cell(
                    _exit_enter_at(dest), f"{ent.id} exit {canon}", "enter_at"
                )
                if enter_cell is not None:
                    ex, ey, ez = enter_cell
                    exit_prop_lines.append(
                        f'engine.set_exit_entry(n_{ent.id}, "{canon}", {ex}, {ey}, {ez})'
                    )
            # Undirected `map` edge (senses + adjacency + "go <room name>"),
            # one per unordered pair, no self-loops.
            if dest_slug == ent.id:
                continue
            pair = frozenset((ent.id, dest_slug))
            if pair in seen_map_pairs:
                continue
            seen_map_pairs.add(pair)
            relation_lines.append(
                f'engine.relate("map", n_{ent.id}, n_{dest_slug})'
            )

    if relation_lines:
        parts.append("-- Relations")
        parts.extend(relation_lines)
        parts.append("")
    if entrance_lines:
        parts.append("-- Room entrances (spawn/arrival cells)")
        parts.extend(entrance_lines)
        parts.append("")
    if exit_prop_lines:
        parts.append("-- Directional exits (exit_<dir> = destination node)")
        parts.extend(exit_prop_lines)
        parts.append("")

    # 5a2. Cell occupancy / blocking layer (§22 Phase 5, #108) ----------------
    # A container's `blocked:` list lowers to engine.set_blocked calls. Default
    # kind is a solid wall (blocks move + sight); `[x,y,z,move|sight]` narrows it.
    blocked_lines: list[str] = []
    for ent in world_entities:
        blocked = ent.properties.get("blocked")
        if blocked is None:
            continue
        for (x, y, z, flags) in _parse_blocked_cells(blocked, ent.id):
            if flags == 3:
                blocked_lines.append(f"engine.set_blocked(n_{ent.id}, {x}, {y}, {z})")
            else:
                blocked_lines.append(
                    f"engine.set_blocked(n_{ent.id}, {x}, {y}, {z}, {flags})"
                )
    if blocked_lines:
        parts.append("-- Cell occupancy / blocking layer (set_blocked)")
        parts.extend(blocked_lines)
        parts.append("")

    # 5b. Entity reaction triggers (non-verb world entities) ------------------
    # Emit engine.set_trigger(n_<id>, "<slot>", function(ctx) ... end) for each
    # lua/luau trigger authored DIRECTLY ON THE INSTANCE.
    #
    # We deliberately do NOT include kind-inherited triggers here. In the graph
    # model, generic per-kind verb behavior is implemented by the migrated graph
    # VERBS (their stages), not by per-entity reactions. The core stdlib kinds
    # still carry legacy verb-handler triggers (container's test:Open, item's
    # on:Take, …) written for the old ctx.self model; emitting those onto every
    # instance makes them fire in the bubble chain with a nil ctx.self and
    # spuriously fail core verbs (e.g. Open → "already open"). Entity reactions
    # are authored on the specific entity (e.g. the Hanged Corpse's on:Answer),
    # which is exactly `ent.triggers`. Kind-level reactions can be revisited once
    # the core kinds are stripped of their legacy handlers (engine task: retire
    # the legacy Luau-dispatch path).
    # In om mode the SAME instance lua/luau triggers lower to Layer-B reactions
    # (SCENES_AND_ACTS §3): om.on(node, "On<Event>", function(ctx) … end), fired
    # by the engine/trampoline event bus. The closure receives the event payload
    # as `ctx`. (NB: bodies authored against the legacy host API — ctx.noun,
    # engine.set_property/set_world — won't resolve against the om reaction ctx +
    # graph store; those bodies need a content port. The LOWERING is correct; the
    # legacy body content is a separate sample-dungeon concern.)
    trigger_lines: list[str] = []
    for ent in world_entities:
        instance_triggers = ent.triggers
        lua_entity_triggers = [
            t for t in instance_triggers
            if t.script is not None and t.script.language in ("lua", "luau")
        ]
        non_lua_entity_triggers = [
            t for t in instance_triggers
            if t.script is None or t.script.language not in ("lua", "luau")
        ]
        for trigger in lua_entity_triggers:
            if om:
                if _verb_stage_key(trigger.name) in _OM_INTERCEPTION_STAGES:
                    # interception → a Before-stage SELF behaviour on the
                    # reach-path (can ctx:veto). Test+InsteadOf for the same verb
                    # both land here; author one Before trigger to avoid the
                    # later-wins collision on (node, verb, before, self).
                    verb = _om_interception_verb(trigger.name)
                    trigger_lines.append(
                        f'wyrd.set_behaviour(n_{ent.id}, {_lua_string(verb)}, "before", "self", function(ctx)'
                    )
                else:
                    event = _om_event_name(trigger.name)
                    trigger_lines.append(
                        f"wyrd.on(n_{ent.id}, {_lua_string(event)}, function(ctx)"
                    )
            else:
                slot = _trigger_slot_key(trigger.name)
                trigger_lines.append(
                    f"engine.set_trigger(n_{ent.id}, {_lua_string(slot)}, function(ctx)"
                )
            guard = _compile_when_guard(trigger.when) if trigger.when else None
            if trigger.when and guard is None:
                trigger_lines.append(
                    f"    -- TODO: when guard not compiled: {trigger.when!r}"
                )
            elif guard is not None:
                trigger_lines.append(f"    if not ({guard}) then return end")
            body = _trigger_body(trigger)
            if body:
                for line in body.splitlines():
                    trigger_lines.append(f"    {line}" if line.strip() else "")
            trigger_lines.append("end)")
        for trigger in non_lua_entity_triggers:
            # Only warn when the trigger has any content at all (body or script).
            if trigger.script is not None or trigger.body:
                trigger_lines.append(
                    f"-- WARNING: entity {_lua_string(ent.id)} trigger {_lua_string(trigger.name)}"
                    f" has a non-script FML body; not transpiled (lean scope)"
                    f" — rewrite as emergent or native lua"
                )
    if trigger_lines:
        header = "Entity reaction triggers (wyrd.on)" if om else "Entity reaction triggers (set_trigger)"
        parts.append(f"-- {header}")
        parts.extend(trigger_lines)
        parts.append("")

    # 6. Verbs ----------------------------------------------------------------
    # stdlib module: all verbs. floor: only floor-local verbs (skip imported).
    verb_entities = [
        e for e in floor.entities.values()
        if e.kind == "verb" and _in_scope(e)
    ]
    # Build a set of understand_directives indexed by verb_id for O(1) lookup.
    ud_by_verb: dict[str, list[str]] = {}
    for directive in floor.understand_directives:
        for phrase in directive.phrases:
            if isinstance(phrase, str) and phrase:
                ud_by_verb.setdefault(directive.verb_id, []).append(phrase)

    for ent in verb_entities:
        verb_name = ent.id
        # Skip built-in verbs.
        if verb_name in _BUILTIN_VERBS:
            continue

        parts.append(f"-- verb: {verb_name}")
        parts.append(f"engine.define_verb({{")
        parts.append(f"    name = {_lua_string(verb_name)},")

        # Grammar fields → engine.define_verb spec, mapping the FML property
        # names (noun_2, noun_scope) onto the spec names the engine expects
        # (noun2, scope). First present name wins. (noun_scope_2 has no spec
        # field — the engine uses one scope hint for both nouns — so it's dropped.)
        grammar_map = (
            ("noun",           ("noun",)),
            ("noun2",          ("noun2", "noun_2")),
            ("preposition",    ("preposition",)),
            ("scope",          ("scope", "noun_scope")),
            ("target_rel",     ("target_rel",)),
            ("subject_is_src", ("subject_is_src",)),
            ("event",          ("event",)),
        )
        for spec_key, fml_names in grammar_map:
            for nm in fml_names:
                val = ent.properties.get(nm)
                if val is not None:
                    parts.append(f"    {_lua_key(spec_key)} = {_lua_value(val)},")
                    break

        # Aliases: from props aliases/phrases + understand_directives.
        alias_list: list[str] = []
        for alias_key in ("aliases", "phrases"):
            alias_val = ent.properties.get(alias_key)
            if alias_val is not None:
                if isinstance(alias_val, str):
                    alias_list.append(alias_val)
                elif isinstance(alias_val, list):
                    alias_list.extend(a for a in alias_val if isinstance(a, str))
        alias_list.extend(ud_by_verb.get(verb_name, []))
        if alias_list:
            items_lua = ", ".join(_lua_string(a) for a in alias_list)
            parts.append(f"    aliases = {{ {items_lua} }},")

        # Stages: only from lua/luau script triggers. In om mode the verb is
        # emitted GRAMMAR-ONLY (behaviour is om.set_behaviour fragments in the
        # deferred behaviour-port phase), so no stages are emitted.
        lua_triggers = [] if om else [
            t for t in ent.triggers
            if t.script is not None and t.script.language in ("lua", "luau")
        ]
        non_lua_triggers = [] if om else [
            t for t in ent.triggers
            if t.script is None or t.script.language not in ("lua", "luau")
        ]

        if lua_triggers:
            parts.append("    stages = {")
            for trigger in lua_triggers:
                stage_key = _verb_stage_key(trigger.name)
                parts.append(f"        {stage_key} = function(ctx)")
                for line in trigger.script.source.splitlines():  # type: ignore[union-attr]
                    parts.append(f"            {line}" if line.strip() else "")
                parts.append("        end,")
            parts.append("    },")

        parts.append("})")

        # Warn about non-lua trigger bodies.
        for trigger in non_lua_triggers:
            parts.append(
                f"-- WARNING: verb {_lua_string(verb_name)} trigger {_lua_string(trigger.name)}"
                f" has a non-script FML body; not transpiled (lean scope)"
                f" — rewrite as emergent or native lua"
            )

        parts.append("")

    # 7. Start actor (floor mode only — a stdlib module has no player) --------
    if not stdlib_module:
        start_id: str | None = None

        # (a) an explicit start-actor entity named by a floor property.
        for prop_key in ("start_actor", "start", "player"):
            val = floor.properties.get(prop_key)
            if isinstance(val, str) and val in world_ids:
                start_id = val
                break

        # (b) an entity whose kind chain includes 'player'/'pc'.
        if start_id is None:
            for ent in world_entities:
                chain = ent.kind_chain or [ent.kind]
                if any(k in ("player", "pc") for k in chain):
                    start_id = ent.id
                    break

        if start_id is not None:
            parts.append(f"engine.set_start_actor(n_{start_id})")
        else:
            # (c) No player entity exists. Full dynamic-loading (engine #102):
            # declare the floor's start ROOM and pre-seed NO actor — every
            # player is spawned at runtime (Loom's spawn_actor / the engine's
            # local-play bootstrap), so the floor seeds no unmanned standin.
            start_room = None
            for prop_key in ("start_location", "start", "start_room"):
                val = floor.properties.get(prop_key)
                if isinstance(val, str) and val in world_ids:
                    start_room = val
                    break
            if start_room is not None:
                parts.append(f"engine.set_start_location(n_{start_room})")
            else:
                parts.append(
                    "-- engine.set_start_actor: no player/start entity determined; set manually"
                )

    parts.append("")
    return "\n".join(parts)


_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]*\)")


def _flatten_prose_markup(s: str) -> str:
    """Reduce inline Markdown to plain text for an in-game description:
    `[north](#Hall)` → `north`. Leaves other text untouched."""
    return _MD_LINK_RE.sub(r"\1", s)


# Movement-direction canonicalization — mirrors canon_direction() in the engine
# (engine-core src/graph_runtime.c). Maps an exit key to the full-word form used
# for the room's exit_<dir> property, so "go north"/"go n"/bare "n" all resolve.
_DIR_CANON = {
    "north": "north", "n": "north", "south": "south", "s": "south",
    "east": "east", "e": "east", "west": "west", "w": "west",
    "up": "up", "u": "up", "down": "down", "d": "down",
    "northeast": "northeast", "ne": "northeast",
    "northwest": "northwest", "nw": "northwest",
    "southeast": "southeast", "se": "southeast",
    "southwest": "southwest", "sw": "southwest",
    "in": "in", "inside": "in", "out": "out", "outside": "out",
}

# A direction key must be a safe `exit_<name>` property-key suffix: lowercase
# alphanumerics + underscore. Author-defined directions that aren't already a
# known alias pass through as-is if they match; anything else (spaces, quotes)
# is skipped for the exit_ property (the undirected `map` edge still connects).
_DIR_KEY_RE = re.compile(r"[a-z0-9_]+")


def _exit_dest_slug(dest: Any) -> str | None:
    """An exit value is either a destination slug (str) or a `{room, door}`
    table (a door exit); return the destination room slug in both cases."""
    if isinstance(dest, str):
        return dest
    if isinstance(dest, dict):
        room = dest.get("room")
        if isinstance(room, str):
            return room
    return None


def _exit_door_slug(dest: Any) -> str | None:
    """The gating door slug of a `{room, door}` exit, or None for a plain exit."""
    if isinstance(dest, dict):
        door = dest.get("door")
        if isinstance(door, str):
            return door
    return None


def _exit_enter_at(dest: Any) -> Any | None:
    """The per-exit `enter_at: [x,y,z]` of an object-form exit, else None."""
    if isinstance(dest, dict):
        return dest.get("enter_at")
    return None


# §22 (Phase 5) — spatial authoring: integer cell positions + occupancy.
#
# Per-axis cell range mirrors the engine's packable range (wyrd src/cell.h:
# 21-bit signed). Validate here so authors get an FML-level error rather than a
# runtime engine rejection.
_CELL_MIN = -1048576
_CELL_MAX = 1048575

# Occupancy block kinds → engine.set_blocked flag bitmask
# (bit 0 = blocks movement, bit 1 = blocks sight / line-of-effect).
_OCC_KIND_FLAGS = {"wall": 3, "move": 1, "sight": 2}


def _is_int(v: Any) -> bool:
    """True for a genuine int (bool is excluded — it is an int subclass)."""
    return isinstance(v, int) and not isinstance(v, bool)


def _coord_in_range(v: int, ent_id: str, what: str) -> int:
    if not (_CELL_MIN <= v <= _CELL_MAX):
        raise FmlSyntaxError(
            f"{what} on {ent_id!r}: cell coordinate {v} out of range "
            f"[{_CELL_MIN}, {_CELL_MAX}]"
        )
    return v


def _parse_cell(value: Any, ent_id: str, what: str = "position") -> tuple[int, int, int] | None:
    """Parse a `[x, y, z]` cell value into a validated integer triple, or None if
    none was authored. Raises FmlSyntaxError on a malformed one. `what` names the
    authoring key for the error message (position / entrance / enter_at)."""
    if value is None:
        return None
    if not (isinstance(value, list) and len(value) == 3 and all(_is_int(v) for v in value)):
        raise FmlSyntaxError(
            f"{what} on {ent_id!r} must be three integers [x, y, z]; got {value!r}"
        )
    return tuple(_coord_in_range(int(v), ent_id, what) for v in value)  # type: ignore[return-value]


def _parse_blocked_cells(value: Any, ent_id: str) -> list[tuple[int, int, int, int]]:
    """Parse a `blocked:` value (a list of cells) into validated
    (x, y, z, flags) tuples. Each cell is `[x, y, z]` (defaults to a solid wall:
    blocks move + sight) or `[x, y, z, kind]` with kind in wall/move/sight.
    Raises FmlSyntaxError on a malformed cell."""
    if not isinstance(value, list):
        raise FmlSyntaxError(
            f"blocked on {ent_id!r} must be a list of cells [[x,y,z], ...]; got {value!r}"
        )
    out: list[tuple[int, int, int, int]] = []
    for entry in value:
        if not (isinstance(entry, list) and 3 <= len(entry) <= 4
                and all(_is_int(v) for v in entry[:3])):
            raise FmlSyntaxError(
                f"blocked cell on {ent_id!r} must be [x, y, z] or [x, y, z, kind] "
                f"with integer coords; got {entry!r}"
            )
        x, y, z = (_coord_in_range(int(entry[i]), ent_id, "blocked") for i in range(3))
        flags = 3
        if len(entry) == 4:
            kind = entry[3]
            if kind not in _OCC_KIND_FLAGS:
                raise FmlSyntaxError(
                    f"blocked cell kind on {ent_id!r} must be one of "
                    f"{'/'.join(sorted(_OCC_KIND_FLAGS))}; got {kind!r}"
                )
            flags = _OCC_KIND_FLAGS[kind]
        out.append((x, y, z, flags))
    return out


def _alias_blob(ent: FMLEntity) -> str | None:
    """An entity's noun aliases as a pipe-delimited, lowercased, pipe-bracketed
    string ("|cup|bone cup|") for the engine's noun resolver, or None if it has
    no aliases. Pipe-bracketing lets the engine test "|<noun>|" for exact match."""
    vals = ent.properties.get("aliases")
    if isinstance(vals, str):
        strs = [vals]
    elif isinstance(vals, list):
        strs = [a for a in vals if isinstance(a, str)]
    else:
        return None
    cleaned = [a.strip().lower() for a in strs if a.strip()]
    if not cleaned:
        return None
    return "|" + "|".join(cleaned) + "|"


def _prose_function_literal(ent: FMLEntity) -> str | None:
    """Lower an entity's prose to a Luau ``function(self, ctx) … end`` literal
    for engine.set_prose, or None if the entity has no prose.

    ProseValue → the full _compile_prose lowering (inline conditionals/
    substitutions preserved, evaluated at output time); markdown links flattened.
    LuauCode → the round-tripped function source verbatim.
    plain str → a trivial function returning the (flattened) text.
    """
    p = ent.prose
    if isinstance(p, ProseValue):
        return _compile_prose(p, "prose", flatten_markdown=True)
    if isinstance(p, LuauCode):
        return p.source
    if isinstance(p, str):
        s = _flatten_prose_markup(p.strip())
        if not s:
            return None
        return f"function(self, ctx) return {_lua_string(s)} end"
    return None


def _entity_description(ent: FMLEntity) -> str | None:
    """Flatten an entity's prose into a single description string, or None.

    Plain-string prose is used as-is; a ProseValue (duck-typed by its `lines`)
    is joined preserving paragraph breaks. Markdown links are reduced to their
    text. Returns None when there is no prose.
    """
    p = ent.prose
    if isinstance(p, str):
        s = p.strip()
    else:
        lines = getattr(p, "lines", None)
        s = "\n".join(lines).strip() if lines else ""
    if not s:
        return None
    return _flatten_prose_markup(s)


def _collect_scalar_props(ent: FMLEntity) -> dict[str, Any]:
    """Return only scalar (str/int/float/bool) properties, excluding location/exits/exit.

    ``name`` is also skipped: it is emitted explicitly as the create_node name,
    and repeating it would produce a duplicate Lua table key.
    """
    _SKIP = frozenset({"name", "location", "exits", "exit", "position", "blocked"})
    result: dict[str, Any] = {}
    for k, v in ent.properties.items():
        if k in _SKIP:
            continue
        # Accept only primitive scalars (bool before int, since bool is subclass of int).
        if isinstance(v, (bool, str, float)) or (isinstance(v, int) and not isinstance(v, bool)):
            result[k] = v
    return result


# ─── Public entry points ──────────────────────────────────────────────────────


def emit_lua_stdlib_module(floor: Floor, source_path: str | None = None) -> str:
    """Emit a stdlib FML floor (verbs.md) as a Lua verb-module string.

    Returns a Lua module that ``return``s a table keyed by verb name.
    Each value is a verb descriptor compatible with the C dispatch engine:
    ``{ name, event, aliases, ..., stages = { test=fn, on=fn, ... } }``.

    Also emits ``engine.register_verb_alias`` calls (before the ``return``)
    for each alias listed in a verb's ``aliases`` property, so that single-
    word shortcuts like ``"n"`` and multi-word phrases like ``"get off"``
    resolve correctly at dispatch time.

    Handler indirection: verbs with a ``handler`` property but no triggers of
    their own (e.g. ``north`` → ``go_handler``) delegate to named handler
    functions emitted at module level.  The canonical implementation is the
    entity whose ``handler`` value matches the handler name AND which also
    carries triggers (e.g. the ``go`` entity).

    Only entities whose ``kind == "verb"`` are emitted.
    """
    src = source_path or "unknown.md"
    parts = [f"-- stdlib verb module — generated from {src}"]

    # --- Pass 1: build handler_map ----------------------------------------
    # handler_name → entity that implements it (has triggers + matching handler)
    handler_map: dict[str, "FMLEntity"] = {}
    for entity in floor.entities.values():
        if entity.kind != "verb":
            continue
        if not entity.triggers:
            continue
        h = entity.properties.get("handler", "")
        if h and h not in handler_map:
            handler_map[h] = entity

    # Collect alias registration calls first; emit before the return table.
    alias_calls: list[str] = []
    for entity in floor.entities.values():
        if entity.kind != "verb":
            continue
        aliases = entity.properties.get("aliases")
        if aliases:
            if isinstance(aliases, str):
                aliases = [aliases]
            if isinstance(aliases, list):
                for phrase in aliases:
                    if isinstance(phrase, str) and phrase:
                        alias_calls.append(
                            f"engine.register_verb_alias("
                            f"{_lua_string(phrase)}, {_lua_string(entity.id)})"
                        )
        phrases = entity.properties.get("phrases")
        if phrases:
            if isinstance(phrases, str):
                phrases = [phrases]
            if isinstance(phrases, list):
                for phrase in phrases:
                    if isinstance(phrase, str) and phrase:
                        alias_calls.append(
                            f"engine.register_verb_alias("
                            f"{_lua_string(phrase)}, {_lua_string(entity.id)})"
                        )

    # Understand directives (floor-level; may carry full-command targets like "go north").
    for directive in floor.understand_directives:
        for phrase in directive.phrases:
            if isinstance(phrase, str) and phrase:
                alias_calls.append(
                    f"engine.register_verb_alias("
                    f"{_lua_string(phrase)}, {_lua_string(directive.verb_id)})"
                )

    if alias_calls:
        parts.append("")
        parts.append("-- verb alias registrations")
        parts.extend(alias_calls)

    # Named handler closures: one local per (handler_name, stage_key) pair.
    handler_fn_names: dict[str, dict[str, str]] = {}  # handler_name → {stage_key → local_name}
    for handler_name, impl_entity in handler_map.items():
        stage_fns: dict[str, str] = {}
        for trigger in impl_entity.triggers:
            stage_key = _verb_stage_key(trigger.name)
            local_name = f"_h_{handler_name}_{stage_key}"
            body = _trigger_body(trigger)
            parts.append(f"\nlocal {local_name} = function(ctx)")
            if body:
                for line in body.splitlines():
                    parts.append(f"    {line}" if line.strip() else "")
            parts.append("end")
            stage_fns[stage_key] = local_name
        handler_fn_names[handler_name] = stage_fns

    parts.append("")
    parts.append("return {")

    for entity in floor.entities.values():
        if entity.kind != "verb":
            continue

        parts.append(f"    [{_lua_string(entity.id)}] = {{")
        parts.append(f"        name = {_lua_string(entity.name)},")

        for k, v in entity.properties.items():
            parts.append(f"        {_lua_key(k)} = {_lua_value(v)},")

        handler_name = entity.properties.get("handler", "")
        use_handler_fns = (
            not entity.triggers
            and handler_name
            and handler_name in handler_fn_names
        )

        parts.append("        stages = {")
        if entity.triggers:
            # Verb defines its own stages inline.
            for trigger in entity.triggers:
                stage_key = _verb_stage_key(trigger.name)
                body = _trigger_body(trigger)
                parts.append(f"            {stage_key} = function(ctx)")
                if body:
                    for line in body.splitlines():
                        parts.append(f"                {line}" if line.strip() else "")
                parts.append("            end,")
        elif use_handler_fns:
            # Delegate to named handler closures.
            for stage_key, local_name in handler_fn_names[handler_name].items():
                parts.append(f"            {stage_key} = {local_name},")
        parts.append("        },")
        parts.append("    },")

    parts.append("}")
    return "\n".join(parts)


def emit_lua(
    floor: Floor,
    source_path: str | None = None,
    version: str = "0.1",
    kept: set[str] | None = None,
) -> str:
    """Emit a Floor model as a Luau LFR file string.

    source_path: original .md source path for the header comment.
    version: tower-lower version string for the header comment.
    kept: optional set of entity ids from ``tree_shake()``; when supplied,
        only entities in ``kept`` are emitted.  Pass ``None`` to emit all
        entities (pre-tree-shake conservative behaviour).
    """
    parts: list[str] = []
    _emit_header(parts, floor, source_path, version)
    _emit_floor_table(parts, floor)
    _emit_entity_finder(parts)
    _emit_verb_aliases(parts, floor)

    # Build the section → [entity] map from the floor, respecting section order.
    # Apply tree-shake filter when a kept set is provided.
    section_locals, entity_section = _build_sections(floor, kept=kept)

    for local_name, entities in section_locals:
        if not entities:
            _emit_empty_section(parts, local_name)
        else:
            _emit_section(parts, local_name, entities, floor)

    # Catch-all for kinds that didn't match any section.
    other = entity_section.get(None, [])
    if other:
        parts.append("-- --- Other ---")
        parts.append("local other = {}")
        for entity in other:
            _emit_entity_table(parts, "other", entity, floor)
        _emit_section_triggers(parts, "other", other, floor)
        parts.append("")

    _emit_return(parts, section_locals, other)
    return "\n".join(parts) + "\n"


# ─── Header ───────────────────────────────────────────────────────────────────


def _emit_header(
    parts: list[str],
    floor: Floor,
    source_path: str | None,
    version: str,
) -> None:
    lua_path = source_path.replace(".md", ".lua") if source_path else "unknown.lua"
    parts.append(f"-- {lua_path}")
    src = source_path or "unknown.md"
    parts.append(f"-- Generated by tower-lower {version} from {src}")
    hash_val = floor.fml_source_hash or "unknown"
    parts.append(f"-- fml-source-hash: {hash_val}")
    parts.append("")


# ─── Floor table ─────────────────────────────────────────────────────────────


def _emit_floor_table(parts: list[str], floor: Floor) -> None:
    parts.append("local floor = {")
    parts.append(f"    name = {_lua_string(floor.name)},")
    # Floor-level properties (excluding any that would collide with name/prose).
    for k, v in floor.properties.items():
        if k in ("name", "prose"):
            continue
        parts.append(f"    {_lua_key(k)} = {_lua_value(v)},")
    if floor.prose:
        if isinstance(floor.prose, ProseValue):
            parts.append(f"    prose = {_compile_prose(floor.prose, 'prose')},")
        elif isinstance(floor.prose, LuauCode):
            parts.append(f"    prose = {floor.prose.source},")
        else:
            parts.append(f"    prose = {_lua_string(floor.prose)},")
    if floor.imports:
        items = ", ".join(_lua_string(i) for i in floor.imports)
        parts.append(f"    imports = {{ {items} }},")
    parts.append("    extra = {},")
    parts.append("}")
    parts.append("")


# ─── Entity finder helper ─────────────────────────────────────────────────────


def _emit_entity_finder(parts: list[str]) -> None:
    """Emit a _find_entity(slug) helper that resolves an FML slug to an integer
    entity_id at runtime.  Triggers that reference named entities (BareLink body
    items) use this to obtain the C-side id for engine.call_trigger.

    The helper memoises results in a module-local table so repeat calls are O(1).
    """
    parts.append("-- --- Entity finder (runtime slug → entity_id) ---")
    parts.append("local _eid_cache = {}")
    parts.append("local function _find_entity(slug)")
    parts.append("    if _eid_cache[slug] then return _eid_cache[slug] end")
    parts.append("    local candidates = engine.entities_in_scope(\"global\", 0)")
    parts.append("    for _, eid in ipairs(candidates) do")
    parts.append("        local e = engine.query_entity(eid)")
    parts.append("        if e and e.id == slug then")
    parts.append("            _eid_cache[slug] = eid")
    parts.append("            return eid")
    parts.append("        end")
    parts.append("    end")
    parts.append("    return nil")
    parts.append("end")
    parts.append("")


# ─── Verb alias registration ─────────────────────────────────────────────────


def _emit_verb_aliases(parts: list[str], floor: Floor) -> None:
    """Emit ``engine.register_verb_alias`` calls for every Understand directive.

    One call per phrase in each directive, in declaration order.  The block is
    omitted entirely when there are no directives — keeping the output clean for
    floors that don't use ``**Understand**``.
    """
    if not floor.understand_directives:
        return
    parts.append("-- --- Verb aliases ---")
    for directive in floor.understand_directives:
        for phrase in directive.phrases:
            parts.append(
                f"engine.register_verb_alias({_lua_string(phrase)}, {_lua_string(directive.verb_id)})"
            )
    parts.append("")


# ─── Section building ─────────────────────────────────────────────────────────


def _build_sections(
    floor: Floor,
    kept: set[str] | None = None,
) -> tuple[list[tuple[str, list[FMLEntity]]], dict[str | None, list[FMLEntity]]]:
    """Return (ordered [(local_name, [entity])], {section|None: [entity]}).

    When floor.section_mappings is populated (from stdlib import), entities are
    routed to the declared sections by kind. Without stdlib section mappings,
    all entities land in a single flat ``entities`` table — no guessing.
    Kind definitions and verb declarations are excluded in either case.

    When ``kept`` is provided (from ``tree_shake()``), only entities whose id
    appears in ``kept`` are included in the output sections.  This filters out
    unreachable stdlib catalog content (monsters, spells, items with no authored
    instance) while preserving all verbs and kind-chain ancestors.
    """
    non_skip = [
        e for e in floor.entities.values()
        if e.kind not in _SKIP_KINDS
        and (kept is None or e.id in kept)
    ]

    if not floor.section_mappings:
        # No stdlib: flat table, no hardcoded routing.
        return [("entities", non_skip)], {"entities": non_skip}

    # stdlib-declared sections: route by kind.
    local_order: list[str] = []
    local_kinds: dict[str, list[str]] = {}
    for heading, kind in floor.section_mappings:
        local_name = heading.lower().replace(" ", "_")
        if local_name not in local_kinds:
            local_order.append(local_name)
            local_kinds[local_name] = []
        local_kinds[local_name].append(kind)

    kind_to_local: dict[str, str] = {}
    for local_name, kinds in local_kinds.items():
        for k in kinds:
            if k not in kind_to_local:
                kind_to_local[k] = local_name

    buckets: dict[str | None, list[FMLEntity]] = {}
    for entity in non_skip:
        local = kind_to_local.get(entity.kind)
        buckets.setdefault(local, []).append(entity)

    ordered_sections = [
        (local_name, buckets.get(local_name, []))
        for local_name in local_order
    ]
    return ordered_sections, buckets


# ─── Section emission ─────────────────────────────────────────────────────────


def _emit_empty_section(parts: list[str], local_name: str) -> None:
    label = local_name.replace("_", " ").title()
    parts.append(f"-- --- {label} ---")
    parts.append(f"local {local_name} = {{}}")
    parts.append("")


def _emit_section(
    parts: list[str], local_name: str, entities: list[FMLEntity], floor: Floor
) -> None:
    label = local_name.replace("_", " ").title()
    parts.append(f"-- --- {label} ---")
    parts.append(f"local {local_name} = {{}}")
    for entity in entities:
        parts.append("")
        _emit_entity_table(parts, local_name, entity, floor)
    _emit_section_triggers(parts, local_name, entities, floor)
    parts.append("")


def _emit_section_triggers(
    parts: list[str], local_name: str, entities: list[FMLEntity], floor: Floor
) -> None:
    for entity in entities:
        _, merged_triggers = _resolve_inherited_properties(entity, floor)
        if merged_triggers:
            for trigger in merged_triggers:
                _emit_trigger_attachment(parts, local_name, entity.id, trigger)
        for sub in entity.subentities:
            for trigger in sub.triggers:
                parts.append(
                    f"-- TODO: wire sub-entity trigger {entity.id}/{sub.id}.{trigger.name}"
                )


# ─── Entity table ─────────────────────────────────────────────────────────────


def _emit_entity_table(
    parts: list[str], local_name: str, entity: FMLEntity, floor: Floor
) -> None:
    # Resolve kind-chain inherited properties before emitting.
    # Returns the full flat property set with instance values winning.
    merged_props, _triggers_unused = _resolve_inherited_properties(entity, floor)

    parts.append(f"{local_name}.{entity.id} = {{")
    parts.append(f"    id = {_lua_string(entity.id)},")
    parts.append(f"    kind = {_lua_string(entity.kind)},")
    parts.append(f"    name = {_lua_string(entity.name)},")

    # exits only for rooms, inlined at top level — use merged exits.
    exits = merged_props.get("exits") or merged_props.get("exit")
    if exits and isinstance(exits, dict):
        parts.append(f"    exits = {_lua_value(exits)},")

    if entity.prose:
        if isinstance(entity.prose, ProseValue):
            parts.append(f"    prose = {_compile_prose(entity.prose, 'prose')},")
        elif isinstance(entity.prose, LuauCode):
            # Round-tripped from lua_reader — emit the function literal verbatim.
            parts.append(f"    prose = {entity.prose.source},")
        else:
            parts.append(f"    prose = {_lua_string(entity.prose)},")

    if entity.links:
        items = ", ".join(_lua_string(link) for link in entity.links)
        parts.append(f"    links = {{ {items} }},")

    # properties dict — emit merged (kind-chain + instance), skipping exits.
    remaining_props = {
        k: v for k, v in merged_props.items()
        if k not in ("exits", "exit")
    }
    if remaining_props:
        parts.append("    properties = {")
        for k, v in remaining_props.items():
            parts.append(f"        {_lua_key(k)} = {_lua_prop_value(v)},")
        parts.append("    },")
    else:
        parts.append("    properties = {},")

    # Sub-entities inline.
    if entity.subentities:
        parts.append("    subentities = {")
        for sub in entity.subentities:
            parts.append("        {")
            parts.append(f"            id = {_lua_string(sub.id)},")
            parts.append(f"            kind = {_lua_string(sub.kind)},")
            parts.append(f"            name = {_lua_string(sub.name)},")
            if sub.prose:
                if isinstance(sub.prose, ProseValue):
                    parts.append(f"            prose = {_compile_prose(sub.prose, 'prose')},")
                elif isinstance(sub.prose, LuauCode):
                    parts.append(f"            prose = {sub.prose.source},")
                else:
                    parts.append(f"            prose = {_lua_string(sub.prose)},")
            sub_props = dict(sub.properties)
            if sub_props:
                parts.append("            properties = {")
                for k, v in sub_props.items():
                    parts.append(f"                {_lua_key(k)} = {_lua_prop_value(v)},")
                parts.append("            },")
            else:
                parts.append("            properties = {},")
            parts.append("        },")
        parts.append("    },")

    parts.append("    triggers = {},")
    parts.append("}")


# ─── Trigger attachments ──────────────────────────────────────────────────────


def _trigger_slot_key(name: str) -> str:
    """Convert 'On Enter' / 'InsteadOf Attack' to 'on:Enter' / 'instead_of:Attack'."""
    parts = name.split(None, 1)
    if not parts:
        return "on:Unknown"
    stage_word = parts[0]
    event = parts[1] if len(parts) > 1 else ""
    stage_prefix = _STAGE_MAP.get(stage_word, stage_word.lower())
    return f"{stage_prefix}:{event}" if event else stage_prefix


def _emit_trigger_attachment(
    parts: list[str], local_name: str, entity_id: str, trigger: Trigger
) -> None:
    slot = _trigger_slot_key(trigger.name)
    parts.append(f'{local_name}.{entity_id}.triggers["{slot}"] = function(ctx)')
    guard = _compile_when_guard(trigger.when) if trigger.when else None
    if trigger.when and guard is None:
        # Complex guard that couldn't be compiled — emit a comment so authors
        # know the guard was dropped. The trigger will fire unconditionally.
        parts.append(f"    -- TODO: when guard not compiled: {trigger.when!r}")
    elif guard is not None:
        # guard is a Lua expression that is TRUE when the trigger SHOULD fire.
        # If it evaluates to false, return early (skip the trigger body).
        parts.append(f"    if not ({guard}) then return end")
    body = _trigger_body(trigger)
    if body:
        for line in body.splitlines():
            parts.append(f"    {line}" if line.strip() else "")
    parts.append("end")


# Pattern: flag(identifier)
_FLAG_RE = re.compile(r"^flag\(([A-Za-z_][A-Za-z0-9_]*)\)$")
# Pattern: not flag(identifier)
_NOT_FLAG_RE = re.compile(r"^not\s+flag\(([A-Za-z_][A-Za-z0-9_]*)\)$")


def _compile_when_guard(when: str) -> str | None:
    """Compile a FML 'when:' guard string to a Lua boolean expression.

    Handles simple flag(x) and not flag(x) forms via engine.get_world.
    Returns None for guards that can't be compiled (leaves trigger unconditional
    with a TODO comment — better to fire than to silently not fire).
    """
    s = when.strip()
    m = _NOT_FLAG_RE.match(s)
    if m:
        flag_name = m.group(1)
        return f'engine.get_world({_lua_string(flag_name)}) ~= "true"'
    m = _FLAG_RE.match(s)
    if m:
        flag_name = m.group(1)
        return f'engine.get_world({_lua_string(flag_name)}) == "true"'
    # Complex guard (e.g. time_in_room predicates) — emit a TODO comment
    # and leave the trigger unconditional for now.
    return None


def _trigger_body(trigger: Trigger) -> str:
    if trigger.script is not None:
        if trigger.script.language == "lua" or trigger.script.language == "luau":
            return trigger.script.source
        # Python script from pre-pivot FML
        return "-- NOTE: original trigger was Python; manual Luau translation required"
    if trigger.body:
        return _compile_body_items(trigger.body)
    return ""


def _compile_body_items(items: list[TriggerBodyItem]) -> str:
    return "\n".join(line for item in items for line in _compile_item(item))


def _compile_item(item: TriggerBodyItem) -> list[str]:
    if isinstance(item, PropertySet):
        segments = item.path.rsplit(".", 1)
        if len(segments) == 2:
            entity_path, prop = segments
            return [f"engine.set_property(ctx.{entity_path}.id, {_lua_string(prop)}, {_lua_value(item.value)})"]
        return [f"-- TODO: malformed property path {item.path!r}"]

    if isinstance(item, OutputLine):
        return [f"engine.output({_template_to_lua(item.template)})"]

    if isinstance(item, BareLink):
        # A bare Markdown link in a trigger body is a reference to another entity.
        # Compile to: look up the entity by its FML slug and fire its On Start trigger.
        # Target is typically "#slug" — strip the leading "#" if present.
        slug = item.target.lstrip("#")
        return [
            f"do",
            f"    local _tgt_id = _find_entity({_lua_string(slug)})",
            f"    if _tgt_id then engine.call_trigger(_tgt_id, \"on:Start\", ctx) end",
            f"end",
        ]

    if isinstance(item, ActionLine):
        return _compile_action_line(item)

    return [f"-- TODO: compile {item.kind!r} body item to Luau"]


# Patterns for Set/Clear [label](flag:name) action lines in trigger bodies.
_SET_FLAG_RE = re.compile(r"^Set\s+\[[^\]]*\]\(flag:([A-Za-z_][A-Za-z0-9_]*)\)\s*$")
_CLEAR_FLAG_RE = re.compile(r"^Clear\s+\[[^\]]*\]\(flag:([A-Za-z_][A-Za-z0-9_]*)\)\s*$")
# Apply [status] for N rounds to [entity](#entity_id) — emit a TODO for now.
_APPLY_STATUS_RE = re.compile(r"^Apply\s+")
# Loop through collection: name in collection — compiled by LoopThroughBlock; ActionLine fallback here.
_CLEAR_REACTIONS_RE = re.compile(r"^Clear reactions from")


def _compile_action_line(item: ActionLine) -> list[str]:
    """Compile a Form A action line to Lua.

    Handles:
    - Set [label](flag:name)  → engine.set_world("name", "true")
    - Clear [label](flag:name) → engine.set_world("name", "false")
    - Other lines            → TODO comment (does not crash)
    """
    raw = item.raw.strip()

    m = _SET_FLAG_RE.match(raw)
    if m:
        flag_name = m.group(1)
        return [f"engine.set_world({_lua_string(flag_name)}, \"true\")"]

    m = _CLEAR_FLAG_RE.match(raw)
    if m:
        flag_name = m.group(1)
        return [f"engine.set_world({_lua_string(flag_name)}, \"false\")"]

    return [f"-- TODO: action line {raw!r}"]


_TEMPLATE_TOKEN_RE = re.compile(r"\*([^*]+)\*|`([^`]+)`")


def _template_to_lua(template: str) -> str:
    """Compile output template to a Lua string expression.

    *path*   → ctx.path         (simple dotted path; assumed string, no tostring)
    `expr`   → tostring(expr)   (arbitrary Luau expression; always wrapped)
    """
    lua_parts: list[str] = []
    last_end = 0
    for m in _TEMPLATE_TOKEN_RE.finditer(template):
        literal = template[last_end:m.start()]
        if literal:
            lua_parts.append(_lua_string(literal))
        if m.group(1) is not None:
            # *path* → ctx.path
            lua_parts.append(f"ctx.{m.group(1)}")
        else:
            # `expr` → tostring(expr)
            lua_parts.append(f"tostring({m.group(2)})")
        last_end = m.end()
    trailing = template[last_end:]
    if trailing:
        lua_parts.append(_lua_string(trailing))
    if not lua_parts:
        return _lua_string("")
    if len(lua_parts) == 1:
        return lua_parts[0]
    return " .. ".join(lua_parts)


# ─── Return statement ─────────────────────────────────────────────────────────


def _emit_return(
    parts: list[str],
    section_locals: list[tuple[str, list[FMLEntity]]],
    other: list[FMLEntity],
) -> None:
    parts.append("return {")
    parts.append("    floor      = floor,")
    for local_name, _ in section_locals:
        parts.append(f"    {local_name:<10} = {local_name},")
    if other:
        parts.append("    other      = other,")
    parts.append("}")


# ─── Prose template compilation ──────────────────────────────────────────────

# Backtick segment tokenizer: splits a prose line into literal and `...` parts.
# Matches the outermost backtick pairs (non-greedy).
_PROSE_BACKTICK_RE = re.compile(r"`([^`]*)`")

# Old conditional syntax — hard error post-implementation (spec §8.1).
_OLD_IF_RE = re.compile(r"\[if\b|\[else\]|\[end if\]", re.IGNORECASE)

# Luau keywords / openers that classify a backtick segment as a statement,
# not a value expression.  We check these after ruling out simple expressions.
# Order matters: longer/more-specific first.
_STMT_OPENERS = (
    "if ",
    "elseif ",
    "else",
    "end",
    "for ",
    "while ",
    "repeat",
    "do",
    "local ",
    "return ",
    "break",
    "continue",
    "function ",
)


def _is_prose_statement(segment: str) -> bool:
    """Return True if a backtick segment is a Luau statement/partial chunk.

    Classification per PROSE.md §3.1:
    - If stripped contents match a known statement opener → statement.
    - Otherwise → expression (emit via tostring()).

    No Luau parser dependency — we use keyword prefix matching which is
    sufficient for the FML prose use cases.
    """
    s = segment.strip()
    if not s:
        return False
    for opener in _STMT_OPENERS:
        if s == opener.rstrip() or s.startswith(opener):
            return True
    return False


def _check_old_if_syntax(text: str, context: str) -> None:
    """Raise FmlSyntaxError if text contains old [if X in Y] conditional syntax.

    Per PROSE.md §8.1: after the prose-template implementation lands, this is
    a hard lower-time error.  Authors must use backtick-embedded Luau instead.
    """
    if _OLD_IF_RE.search(text):
        raise FmlSyntaxError(
            f"Old conditional prose syntax '[if ...]' found in {context}. "
            "Migrate to backtick-embedded Luau per docs/design/PROSE.md §8.2. "
            "Example: '[if X in Y]text[end if]' → "
            "'`if engine.entity_at(self.entity_id, \"X\") then` text `end`'"
        )


def _compile_prose(
    prose_val: "ProseValue",
    prop_key: str = "prose",
    source_path: str = "",
    flatten_markdown: bool = False,
) -> str:
    """Lower a ProseValue to a Luau ``function(self, ctx) ... end`` string.

    Implements the lowering algorithm from docs/design/PROSE.md §4:

    1. Split into ``>`` lines (already done — ``prose_val.lines``).
    2. Tokenize each line into literal + backtick segments.
    3. Backtick segment: statement → emit verbatim; expression → ``tostring()``.
    4. Between adjacent ``>`` lines: emit ``s = s .. " "`` (line-join).
    5. Wrap in ``function(self, ctx) local s = "" ... return s end``.
    6. Prepend debug comment (spec §4.3).

    Returns the complete function literal as a string (no trailing newline).
    The caller embeds it at the property slot in the LFR table.
    """
    lines = prose_val.lines
    src_path = source_path or prose_val.source_path or "unknown"
    start_line = prose_val.source_line

    # Hard-error on old conditional syntax in every line.
    for line in lines:
        _check_old_if_syntax(
            line,
            context=f"{src_path}:{start_line} property {prop_key!r}",
        )

    parts: list[str] = []

    # Debug comment: inline with function opener so 'key = function(self, ctx)' is
    # a single token run in the LFR (required by tests and valid Lua).
    parts.append(
        f"function(self, ctx) -- source: {src_path}:{start_line} (prose template start)"
    )
    parts.append("  local s = \"\"")

    for line_idx, line in enumerate(lines):
        # Inter-line join: emit a space between consecutive non-empty > lines.
        # Empty lines are paragraph separators — emit a double-newline instead.
        if line_idx > 0:
            prev_empty = not lines[line_idx - 1].strip()
            curr_empty = not line.strip()
            if prev_empty or curr_empty:
                # Paragraph separator: an empty line between > blocks marks a
                # paragraph break.  Emit "\n\n" when transitioning from an
                # empty separator line back to content (prev_empty + curr not
                # empty).  Skip when the current line is itself empty.
                if prev_empty and not curr_empty:
                    parts.append("  s = s .. \"\\n\\n\" -- paragraph break")
            else:
                parts.append("  s = s .. \" \" -- line join")

        # Skip empty separator lines entirely (no code emitted for them).
        if not line.strip():
            continue

        line_num = start_line + line_idx + 1  # +1 for the property header
        stripped = line.rstrip()

        # Tokenize into alternating literals and backtick segments.
        cursor = 0
        for m in _PROSE_BACKTICK_RE.finditer(stripped):
            # Literal before this backtick segment.
            literal = stripped[cursor : m.start()]
            if literal:
                if flatten_markdown:
                    literal = _flatten_prose_markup(literal)
                escaped = literal.replace("\\", "\\\\").replace('"', '\\"')
                parts.append(
                    f"  s = s .. \"{escaped}\" -- source: {src_path}:{line_num}"
                )

            segment = m.group(1)
            if _is_prose_statement(segment):
                # Statement/partial chunk — emit verbatim.
                parts.append(
                    f"  {segment} -- source: {src_path}:{line_num}"
                )
            else:
                # Expression — wrap in tostring().
                parts.append(
                    f"  s = s .. tostring({segment}) -- source: {src_path}:{line_num}"
                )

            cursor = m.end()

        # Trailing literal after the last backtick (or the whole line if no backticks).
        trailing = stripped[cursor:]
        if trailing:
            if flatten_markdown:
                trailing = _flatten_prose_markup(trailing)
            escaped = trailing.replace("\\", "\\\\").replace('"', '\\"')
            parts.append(
                f"  s = s .. \"{escaped}\" -- source: {src_path}:{line_num}"
            )

    parts.append("  return s")
    parts.append("end")
    return "\n".join(parts)


# ─── Lua value serialisation ──────────────────────────────────────────────────


def _lua_string(s: str) -> str:
    if "\n" in s and "]]" not in s and not s.endswith("]"):
        return f"[[{s}]]"
    s = s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    return f'"{s}"'


def _lua_key(k: str) -> str:
    if _LUA_IDENT_RE.fullmatch(k):
        return k
    return f'["{k}"]'


def _lua_value(v: Any) -> str:
    """Serialize an arbitrary Python value to a Lua literal."""
    if v is None:
        return "nil"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return repr(v)
    if isinstance(v, LuauCode):
        return v.source
    if isinstance(v, ProseValue):
        return _compile_prose(v)
    if isinstance(v, str):
        return _lua_string(v)
    if isinstance(v, list):
        if not v:
            return "{}"
        items = ", ".join(_lua_value(i) for i in v)
        return f"{{ {items} }}"
    if isinstance(v, dict):
        if not v:
            return "{}"
        pairs = ", ".join(f"{_lua_key(k)} = {_lua_value(val)}" for k, val in v.items())
        return f"{{ {pairs} }}"
    if isinstance(v, Predicate):
        return _lua_string(v.model_dump_json())
    return _lua_string(str(v))


def _lua_prop_value(v: Any) -> str:
    """Like _lua_value but handles Predicate, LuauCode, and ProseValue types.

    - ProseValue: compile to Luau function via _compile_prose().
    - LuauCode: emit the source verbatim (no quoting) so live Luau code
      lands at the property slot in the LFR file.
    - Predicate: serialise as a JSON string (existing behaviour).
    - All other types: delegate to _lua_value.
    """
    if isinstance(v, ProseValue):
        return _compile_prose(v)
    if isinstance(v, LuauCode):
        return v.source
    if isinstance(v, Predicate):
        return _lua_string(v.model_dump_json())
    return _lua_value(v)
