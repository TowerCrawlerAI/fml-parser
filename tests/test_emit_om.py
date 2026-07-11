"""Tests for the P6 object-model emitter (emit_lua_om).

emit_lua_om produces the prototype/fragment LFR for the crawler `--engine om`
dispatch path. It is a thin wrapper over emit_lua_graph(om=True): same DB-init
shape, but each world entity gets a prototype EDGE from its kind (resolved by
name from the `kind:<name>` registry on `object`), the string `kind` property is
dropped, verbs are grammar-only, and behaviour triggers are skipped (the
behaviour port is a later phase). These tests verify that contract — and that
om=False (graph mode) output is unchanged.
"""

from __future__ import annotations

from fml_parser.emit_lua import emit_lua_graph, emit_lua_om
from fml_parser.models import FMLEntity, Floor, Script, Trigger


def make_floor(entities: list[FMLEntity], **floor_props) -> Floor:
    f = Floor(name="Test Floor", properties=floor_props)
    for ent in entities:
        f += ent
    return f


def make_entity(eid, name, kind, properties=None, triggers=None, kind_chain=None):
    return FMLEntity(
        id=eid, name=name, kind=kind,
        properties=properties or {}, triggers=triggers or [],
        kind_chain=kind_chain or [kind],
    )


def _tiny_floor() -> Floor:
    cell = make_entity("cell", "Cell", "room",
                       properties={"exits": {"north": "hall"}})
    hall = make_entity("hall", "Hall", "room",
                       properties={"exits": {"south": "cell"}})
    key = make_entity("key", "Key", "item",
                      properties={"location": "cell"})
    grate = make_entity("grate", "Grate", "scenery",
                        properties={"location": "cell"})
    return make_floor([cell, hall, key, grate], start_location="cell")


# ─── prototype edges ─────────────────────────────────────────────────────────


def test_emits_proto_helper_and_edges():
    out = emit_lua_om(_tiny_floor(), source_path="floor.md")
    # the _proto helper is defined once, resolving via the wyrd.named registry
    assert "local function _proto(n, name)" in out
    assert "local k = wyrd.named(name)" in out
    assert "wyrd.set_prototype(n, k)" in out
    # each world entity gets a prototype edge from its declared name
    assert '_proto(n_key, "item")' in out
    assert '_proto(n_grate, "scenery")' in out
    assert '_proto(n_cell, "room")' in out


def test_drops_string_kind_property():
    # an entity that declared `kind` inline must NOT emit it as a node property
    # (the prototype edge replaces it).
    item = make_entity("sigil", "Sigil", "item",
                       properties={"location": "cell", "kind": "item", "type": "knowledge"})
    cell = make_entity("cell", "Cell", "room")
    out = emit_lua_om(make_floor([cell, item], start_location="cell"))
    assert "kind =" not in out          # no string kind prop anywhere
    assert '_proto(n_sigil, "item")' in out
    assert 'type = "knowledge"' in out  # other props survive


def test_no_player_entity_emits_start_location_not_a_standin():
    # Full dynamic-loading (engine #102): a floor with a start_location but no
    # PC declares the start ROOM and pre-seeds NO actor. The old behaviour
    # synthesized a "you" player node, which lingered as an unmanned standin in
    # multiplayer — that must NOT be emitted anymore.
    out = emit_lua_om(_tiny_floor())
    assert "engine.set_start_location(n_cell)" in out
    assert "set_start_actor" not in out
    assert 'name = "you"' not in out
    assert "n__player" not in out


# ─── verbs are grammar-only; behaviours skipped ──────────────────────────────


def test_verbs_grammar_only_no_stages():
    verb = make_entity(
        "drink", "drink", "verb",
        properties={"noun": "required", "scope": "touch"},
        triggers=[Trigger(name="On", script=Script(language="lua",
                                                   source="engine.output('glug')"))],
    )
    cell = make_entity("cell", "Cell", "room")
    out = emit_lua_om(make_floor([cell, verb], start_location="cell"))
    assert "engine.define_verb({" in out
    assert 'name = "drink"' in out
    assert 'noun = "required"' in out
    # the On stage closure must NOT be emitted (grammar-only)
    assert "stages = {" not in out
    assert "glug" not in out


def test_entity_triggers_lower_to_om_on():
    # a lua/luau instance trigger lowers to om.on(node, "On<Event>", fn) — NOT
    # engine.set_trigger (that's the graph path) — with the body preserved.
    npc = make_entity(
        "oracle", "Oracle", "npc",
        properties={"location": "cell"},
        triggers=[Trigger(name="On Answer", script=Script(language="lua",
                                                          source="engine.output('hmm')"))],
    )
    cell = make_entity("cell", "Cell", "room")
    out = emit_lua_om(make_floor([cell, npc], start_location="cell"))
    assert "engine.set_trigger" not in out          # not the graph path
    assert 'wyrd.on(n_oracle, "OnAnswer", function(ctx)' in out
    assert "engine.output('hmm')" in out            # body preserved
    assert '_proto(n_oracle, "npc")' in out          # still placed structurally


def test_om_registers_world_entities_by_name():
    # Every world entity is registered under its display name so runtime content
    # can resolve it with wyrd.named("<Name>") — e.g. cloning a monster
    # prototype at ambush time: wyrd.create(wyrd.named("Skeleton Scavenger")).
    # Without this, wyrd.named() only sees stdlib kinds and a runtime clone gets
    # a stat-less bare node.
    scav = make_entity("skeleton_scavenger", "Skeleton Scavenger", "npc",
                       properties={"hp": 13, "ac": 13})
    cell = make_entity("cell", "Cell", "room")
    out = emit_lua_om(make_floor([cell, scav], start_location="cell"))
    assert 'wyrd.register("Skeleton Scavenger", n_skeleton_scavenger)' in out
    assert 'wyrd.register("Cell", n_cell)' in out
    # The node it points at is the one carrying the stat block, so a clone of it
    # inherits hp/ac (the whole point — a real combatant, not a bare node).
    assert ('local n_skeleton_scavenger = engine.create_node({ name = '
            '"Skeleton Scavenger"') in out
    # Registration must come AFTER the node is declared (forward-reference to
    # n_<id> would be a nil global under Luau).
    assert (out.index('local n_skeleton_scavenger =')
            < out.index('wyrd.register("Skeleton Scavenger"'))


def test_graph_path_does_not_register_by_name():
    # The name registry is an om-path affordance (it mirrors the om _proto
    # helper). The legacy graph emitter must not emit it.
    scav = make_entity("skeleton_scavenger", "Skeleton Scavenger", "npc",
                       properties={"hp": 13})
    cell = make_entity("cell", "Cell", "room")
    out = emit_lua_graph(make_floor([cell, scav], start_location="cell"))
    assert "wyrd.register(" not in out


def test_om_event_name_mapping():
    # the <stage> <Event> heading collapses to On<Event> for the om event bus.
    from fml_parser.emit_lua import _om_event_name
    assert _om_event_name("On Open") == "OnOpen"
    assert _om_event_name("After Enter") == "OnEnter"
    assert _om_event_name("On RoundStart") == "OnRoundStart"
    assert _om_event_name("InsteadOf Damaged") == "OnDamaged"
    assert _om_event_name("On Answer") == "OnAnswer"
    assert _om_event_name("Look") == "OnLook"          # single non-stage word
    assert _om_event_name("On") == "On"                # bare stage word, no event
    assert _om_event_name("") == "On"                  # empty


def test_om_trigger_when_guard_and_multiple():
    # an om trigger with a `when:` guard emits the guard, and two triggers on
    # one entity each emit their own om.on.
    npc = make_entity(
        "warden", "Warden", "npc",
        properties={"location": "cell"},
        triggers=[
            Trigger(name="On Enter", when="flag(alarm)",
                    script=Script(language="lua", source="engine.output('halt')")),
            Trigger(name="On Take", script=Script(language="lua",
                                                  source="engine.output('mine')")),
        ],
    )
    cell = make_entity("cell", "Cell", "room")
    out = emit_lua_om(make_floor([cell, npc], start_location="cell"))
    assert 'wyrd.on(n_warden, "OnEnter", function(ctx)' in out
    assert 'wyrd.on(n_warden, "OnTake", function(ctx)' in out
    # the when-guard is emitted as an early-return (same form as graph mode)
    assert "then return end" in out


def test_interception_triggers_lower_to_before_self():
    # Test/InsteadOf/Before lua triggers BLOCK an action → a Before-stage SELF
    # behaviour on the reach-path (not a post-commit om.on reaction).
    box = make_entity(
        "chest", "Chest", "container",
        properties={"location": "cell"},
        triggers=[
            Trigger(name="Test Take", script=Script(language="lua",
                    source="if self.open ~= true then ctx:veto('shut') end")),
            Trigger(name="Before Attack", script=Script(language="lua",
                    source="ctx:veto('warded')")),
        ],
    )
    cell = make_entity("cell", "Cell", "room")
    out = emit_lua_om(make_floor([cell, box], start_location="cell"))
    assert 'wyrd.set_behaviour(n_chest, "take", "before", "self", function(ctx)' in out
    assert 'wyrd.set_behaviour(n_chest, "attack", "before", "self", function(ctx)' in out
    assert "wyrd.on(n_chest" not in out          # interception ≠ reaction
    assert "ctx:veto('shut')" in out


def test_reaction_vs_interception_split():
    # On/After → om.on reaction; Test/InsteadOf/Before → before/self behaviour.
    from fml_parser.emit_lua import _om_interception_verb, _OM_INTERCEPTION_STAGES, _verb_stage_key
    assert _om_interception_verb("Test Take") == "take"
    assert _om_interception_verb("InsteadOf Attack") == "attack"
    assert _om_interception_verb("Before Put In") == "put-in"
    assert _verb_stage_key("On Open") not in _OM_INTERCEPTION_STAGES
    assert _verb_stage_key("After Enter") not in _OM_INTERCEPTION_STAGES
    assert _verb_stage_key("InsteadOf Take") in _OM_INTERCEPTION_STAGES


def test_non_lua_trigger_body_warned_not_emitted():
    # an FML action-vocabulary body (no lua/luau script) is not transpiled (same
    # as graph mode) — it warns rather than emitting a broken om.on.
    room = make_entity(
        "hall", "Hall", "room",
        triggers=[Trigger(name="After Enter",
                          script=Script(language="python", source="set_flag('x')"))],
    )
    out = emit_lua_om(make_floor([room], start_location="hall"))
    assert "wyrd.on(n_hall" not in out
    assert "WARNING" in out


# ─── containment + exits preserved ───────────────────────────────────────────


def test_containment_and_exits_preserved():
    out = emit_lua_om(_tiny_floor())
    assert 'engine.relate("in", n_key, n_cell)' in out
    assert 'engine.set_prop(n_cell, "exit_north", n_hall)' in out
    assert 'engine.set_prop(n_hall, "exit_south", n_cell)' in out


# ─── determinism + graph-mode non-regression ─────────────────────────────────


def test_deterministic():
    f = _tiny_floor()
    assert emit_lua_om(f) == emit_lua_om(f)


def test_graph_mode_unchanged_by_om_param():
    # om=False must be byte-identical to the legacy graph emitter default.
    f = _tiny_floor()
    assert emit_lua_graph(f, source_path="x.md") == emit_lua_graph(
        f, source_path="x.md", om=False
    )
    # and the om output must differ (proves the branch is active)
    assert emit_lua_om(f, source_path="x.md") != emit_lua_graph(f, source_path="x.md")
