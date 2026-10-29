import numpy as np

from monitor.detectors import DetectorConfig, Event, FailureDetector
from monitor.recovery import Phase, Recovery
from monitor.wrapper import MonitorConfig, MonitoredController

CFG = DetectorConfig(warmup_steps=0, settle_steps=3, stall_window=10)
CLOSE = np.array([0, 0, 0, 0, 0, 0, 1.0])
OPEN = np.array([0, 0, 0, 0, 0, 0, -1.0])


def q(width):
    return np.array([width / 2, -width / 2])


def moving(t):
    return np.array([0.01 * t, 0.0, 1.0])


def run(det, widths, action, start=0):
    evs = []
    for i, w in enumerate(widths):
        evs.append(det.step(moving(start + i), q(w), action))
    return evs


def test_grasp_miss_when_fingers_close_on_nothing():
    det = FailureDetector(CFG)
    evs = run(det, [0.08, 0.05, 0.02, 0.0018, 0.0018, 0.0018], CLOSE)
    assert Event.GRASP_MISS in evs
    assert Event.DROP not in evs


def test_no_event_when_holding_object():
    det = FailureDetector(CFG)
    evs = run(det, [0.08, 0.05, 0.03, 0.03, 0.03, 0.03, 0.03], CLOSE)
    assert set(evs) == {Event.NONE}
    assert det.holding


def test_drop_when_held_object_slips_out():
    det = FailureDetector(CFG)
    run(det, [0.08, 0.05, 0.03, 0.03, 0.03, 0.03], CLOSE)
    evs = run(det, [0.015, 0.0018, 0.0018, 0.0018], CLOSE, start=6)
    assert Event.DROP in evs


def test_no_l1_event_while_gripper_open():
    det = FailureDetector(CFG)
    evs = run(det, [0.0018] * 8, OPEN)
    assert set(evs) == {Event.NONE}


def test_stall_when_eef_does_not_move():
    det = FailureDetector(CFG)
    evs = [det.step(np.array([0.1, 0.0, 1.0]), q(0.08), OPEN) for _ in range(12)]
    assert Event.STALL in evs


def test_recovery_lifts_and_shifts_then_finishes():
    rec = Recovery(rng=np.random.default_rng(0))
    pos = np.array([0.0, 0.0, 0.9])
    rec.start(pos)
    phases = []
    for _ in range(200):
        if not rec.active:
            break
        a = rec.act(pos)
        assert a[-1] < 0  # 回復中はずっと開いている
        pos = pos + a[:3] * 0.05 * 0.5  # 目標の半分だけ進むロボットの近似
        phases.append(rec.phase)
    assert not rec.active
    assert Phase.LIFT in phases and Phase.SHIFT in phases
    assert pos[2] > 0.9 + 0.07
    assert 0.005 < np.linalg.norm(pos[:2]) < 0.035


def test_controller_recovers_then_asks_for_help():
    calls = {"policy": 0, "reset": 0}

    def policy(_obs):
        calls["policy"] += 1
        return CLOSE

    def reset():
        calls["reset"] += 1

    ctrl = MonitoredController(policy, reset, MonitorConfig(max_retries=1), det_cfg=CFG)
    statuses = []
    pos = np.array([0.0, 0.0, 0.9])
    for _ in range(300):
        a, info = ctrl.act(pos, q(0.0018), None)  # 何度閉じても空振りする
        pos = pos + a[:3] * 0.025
        statuses.append(info["status"])
        if info["help"]:
            break
    assert "GRASP_MISS" in statuses and "RECOVERING" in statuses
    assert statuses[-1] == "HELP"
    assert ctrl.retries == 2
    assert calls["reset"] >= 2  # 開始時 + 回復の完了時に VLA のキューを捨てている


def test_shadow_mode_never_intervenes():
    ctrl = MonitoredController(lambda o: CLOSE, lambda: None, MonitorConfig(enabled=False), det_cfg=CFG)
    events = []
    for i in range(30):
        a, info = ctrl.act(moving(i), q(0.0018), None)
        assert np.allclose(a, CLOSE)
        events.append(info["event"])
    assert events.count("GRASP_MISS") == 1  # 続く間は 1 回だけ数える
