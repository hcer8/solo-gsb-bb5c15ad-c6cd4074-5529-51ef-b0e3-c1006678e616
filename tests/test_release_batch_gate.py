"""POST /release-batch-gate 分批发布门禁单元测试（仅依赖标准库 + 本项目模块）。

运行：python -m unittest discover -s tests -v
"""

from __future__ import annotations

import unittest

from app.serial import (
    HALF,
    MAX_SERIAL,
    BLOCKED,
    DEPENDENCY_BLOCKED,
    READY,
    TARGET_NOT_NEWER,
    ReleaseBatchError,
    evaluate_zone_release,
    plan_release_batches,
)


def zone(name, target, replicas, depends_on=()):
    """构造 plan_release_batches 的输入元组。"""
    return (name, target, list(replicas), list(depends_on))


# ---------------------------------------------------------------------------
# 单 zone 串行评估
# ---------------------------------------------------------------------------


class EvaluateZoneReleaseTests(unittest.TestCase):
    def test_all_newer(self):
        items, max_distance, blocking = evaluate_zone_release(
            "example.com", 105, [("ns-a", 100), ("ns-b", 104)]
        )
        self.assertEqual(
            items,
            [
                ("ns-a", 100, "newer", 5, False),
                ("ns-b", 104, "newer", 1, False),
            ],
        )
        self.assertEqual(max_distance, 5)
        self.assertEqual(blocking, [])

    def test_equal_is_not_blocking(self):
        # 已在目标值的副本无需推进，不阻塞
        items, max_distance, blocking = evaluate_zone_release(
            "example.com", 100, [("ns-a", 100), ("ns-b", 90)]
        )
        self.assertEqual(items[0], ("ns-a", 100, "equal", 0, False))
        self.assertEqual(max_distance, 10)
        self.assertEqual(blocking, [])

    def test_older_is_blocking(self):
        items, _max_distance, blocking = evaluate_zone_release(
            "example.com", 100, [("ns-a", 101), ("ns-b", 99)]
        )
        self.assertEqual(items[0][2], "older")
        self.assertTrue(items[0][4])
        self.assertEqual(items[1], ("ns-b", 99, "newer", 1, False))
        self.assertEqual(blocking, ["ns-a"])

    def test_wraparound_advance_is_newer(self):
        items, max_distance, blocking = evaluate_zone_release(
            "wrap.example", 1, [("ns-a", MAX_SERIAL), ("ns-b", MAX_SERIAL - 1)]
        )
        self.assertEqual(items[0][2], "newer")
        self.assertEqual(items[0][3], 2)  # 4294967295 -> 1 跨回绕点距离为 2
        self.assertEqual(max_distance, 3)
        self.assertEqual(blocking, [])

    def test_ambiguous_raises_structured_error(self):
        with self.assertRaises(ReleaseBatchError) as ctx:
            evaluate_zone_release(
                "example.com", HALF, [("ns-a", 0)]
            )
        detail = ctx.exception.detail
        self.assertEqual(detail["error"], "serial_incomparable")
        self.assertEqual(detail["zone"], "example.com")
        self.assertEqual(detail["replica"], "ns-a")
        self.assertEqual(detail["current"], 0)
        self.assertEqual(detail["target"], HALF)
        self.assertEqual(detail["forward_distance"], HALF)


# ---------------------------------------------------------------------------
# 整体校验（重复 zone / 未知依赖 / 依赖环）
# ---------------------------------------------------------------------------


class PlanReleaseBatchesValidationTests(unittest.TestCase):
    def test_duplicate_zone_name(self):
        with self.assertRaises(ReleaseBatchError) as ctx:
            plan_release_batches([
                zone("a", 1, [("ns-a", 0)]),
                zone("a", 2, [("ns-b", 0)]),
            ])
        detail = ctx.exception.detail
        self.assertEqual(detail["error"], "duplicate_zone_name")
        self.assertEqual(detail["name"], "a")

    def test_unknown_dependency(self):
        with self.assertRaises(ReleaseBatchError) as ctx:
            plan_release_batches([
                zone("a", 1, [("ns-a", 0)], depends_on=["missing"]),
            ])
        detail = ctx.exception.detail
        self.assertEqual(detail["error"], "unknown_dependency")
        self.assertEqual(detail["zone"], "a")
        self.assertEqual(detail["dependency"], "missing")

    def test_self_dependency_is_cycle(self):
        with self.assertRaises(ReleaseBatchError) as ctx:
            plan_release_batches([
                zone("a", 1, [("ns-a", 0)], depends_on=["a"]),
            ])
        detail = ctx.exception.detail
        self.assertEqual(detail["error"], "dependency_cycle")
        self.assertEqual(detail["cycle"], ["a"])

    def test_two_node_cycle_normalized(self):
        with self.assertRaises(ReleaseBatchError) as ctx:
            plan_release_batches([
                zone("b", 1, [("ns-b", 0)], depends_on=["a"]),
                zone("a", 1, [("ns-a", 0)], depends_on=["b"]),
            ])
        detail = ctx.exception.detail
        self.assertEqual(detail["error"], "dependency_cycle")
        # 环规范化为从字典序最小节点开始
        self.assertEqual(detail["cycle"], ["a", "b"])

    def test_three_node_cycle(self):
        with self.assertRaises(ReleaseBatchError) as ctx:
            plan_release_batches([
                zone("c", 1, [("ns-c", 0)], depends_on=["b"]),
                zone("a", 1, [("ns-a", 0)], depends_on=["c"]),
                zone("b", 1, [("ns-b", 0)], depends_on=["a"]),
            ])
        self.assertEqual(ctx.exception.detail["cycle"], ["a", "c", "b"])

    def test_cycle_detection_ignores_acyclic_tail(self):
        # 环外节点（依赖环的下游）不影响环的提取
        with self.assertRaises(ReleaseBatchError) as ctx:
            plan_release_batches([
                zone("tail", 1, [("ns-t", 0)], depends_on=["x"]),
                zone("x", 1, [("ns-x", 0)], depends_on=["y"]),
                zone("y", 1, [("ns-y", 0)], depends_on=["x"]),
            ])
        self.assertEqual(ctx.exception.detail["cycle"], ["x", "y"])

    def test_serial_incomparable_via_plan(self):
        with self.assertRaises(ReleaseBatchError) as ctx:
            plan_release_batches([
                zone("a", HALF, [("ns-a", 0)]),
            ])
        self.assertEqual(ctx.exception.detail["error"], "serial_incomparable")


# ---------------------------------------------------------------------------
# 分批
# ---------------------------------------------------------------------------


class PlanReleaseBatchesTests(unittest.TestCase):
    def test_no_dependencies_single_batch_sorted(self):
        batches, results = plan_release_batches([
            zone("b", 10, [("ns-b", 1)]),
            zone("a", 10, [("ns-a", 1)]),
        ])
        self.assertEqual(batches, [["a", "b"]])
        self.assertTrue(all(r[0] == READY for r in results))
        self.assertTrue(all(r[2] == 0 for r in results))

    def test_chain_layering(self):
        batches, results = plan_release_batches([
            zone("c", 10, [("ns-c", 1)], depends_on=["b"]),
            zone("a", 10, [("ns-a", 1)]),
            zone("b", 10, [("ns-b", 1)], depends_on=["a"]),
        ])
        self.assertEqual(batches, [["a"], ["b"], ["c"]])
        by_name = {name: r for name, r in zip(["c", "a", "b"], results)}
        self.assertEqual(by_name["a"][2], 0)
        self.assertEqual(by_name["b"][2], 1)
        self.assertEqual(by_name["c"][2], 2)

    def test_diamond(self):
        batches, _results = plan_release_batches([
            zone("d", 10, [("ns-d", 1)], depends_on=["b", "c"]),
            zone("b", 10, [("ns-b", 1)], depends_on=["a"]),
            zone("c", 10, [("ns-c", 1)], depends_on=["a"]),
            zone("a", 10, [("ns-a", 1)]),
        ])
        self.assertEqual(batches, [["a"], ["b", "c"], ["d"]])

    def test_batching_is_order_independent(self):
        # 请求顺序不影响分批结果（批内按名称升序）
        forward, _ = plan_release_batches([
            zone("a", 10, [("ns-a", 1)]),
            zone("b", 10, [("ns-b", 1)], depends_on=["a"]),
            zone("c", 10, [("ns-c", 1)], depends_on=["a"]),
        ])
        shuffled, _ = plan_release_batches([
            zone("c", 10, [("ns-c", 1)], depends_on=["a"]),
            zone("b", 10, [("ns-b", 1)], depends_on=["a"]),
            zone("a", 10, [("ns-a", 1)]),
        ])
        self.assertEqual(forward, shuffled)

    def test_serial_blocked_zone_not_scheduled(self):
        batches, results = plan_release_batches([
            zone("a", 10, [("ns-a", 20)]),  # older -> target_not_newer
            zone("b", 10, [("ns-b", 1)]),
        ])
        self.assertEqual(batches, [["b"]])
        status, max_distance, batch, reasons, blocking_replicas, blocking_deps, _ = (
            results[0]
        )
        self.assertEqual(status, BLOCKED)
        self.assertIsNone(batch)
        self.assertEqual(reasons, [TARGET_NOT_NEWER])
        self.assertEqual(blocking_replicas, ["ns-a"])
        self.assertEqual(blocking_deps, [])
        self.assertEqual(max_distance, (10 - 20) % (2 ** 32))

    def test_dependency_blocked_propagates(self):
        batches, results = plan_release_batches([
            zone("a", 10, [("ns-a", 20)]),          # 串行受阻
            zone("b", 10, [("ns-b", 1)], depends_on=["a"]),
            zone("c", 10, [("ns-c", 1)], depends_on=["b"]),
        ])
        self.assertEqual(batches, [])
        self.assertEqual(results[0][3], [TARGET_NOT_NEWER])
        # b 串行可行但依赖未排批的 a
        self.assertEqual(results[1][0], BLOCKED)
        self.assertEqual(results[1][3], [DEPENDENCY_BLOCKED])
        self.assertEqual(results[1][5], ["a"])
        # c 的阻塞沿依赖链传递
        self.assertEqual(results[2][3], [DEPENDENCY_BLOCKED])
        self.assertEqual(results[2][5], ["b"])

    def test_both_reasons_together(self):
        _batches, results = plan_release_batches([
            zone("a", 10, [("ns-a", 20)]),  # target_not_newer
            zone("b", 10, [("ns-b", 30)], depends_on=["a"]),  # 两者皆占
        ])
        self.assertEqual(
            results[1][3], [TARGET_NOT_NEWER, DEPENDENCY_BLOCKED]
        )
        self.assertEqual(results[1][4], ["ns-b"])
        self.assertEqual(results[1][5], ["a"])

    def test_up_to_date_zone_scheduled_with_zero_distance(self):
        batches, results = plan_release_batches([
            zone("a", 10, [("ns-a", 10)]),
            zone("b", 10, [("ns-b", 1)], depends_on=["a"]),
        ])
        self.assertEqual(batches, [["a"], ["b"]])
        self.assertEqual(results[0][0], READY)
        self.assertEqual(results[0][1], 0)  # max_forward_distance

    def test_result_items_shape(self):
        _batches, results = plan_release_batches([
            zone("a", 105, [("ns-a", 100), ("ns-b", 105)]),
        ])
        items = results[0][6]
        self.assertEqual(
            items,
            [
                ("ns-a", 100, "newer", 5, False),
                ("ns-b", 105, "equal", 0, False),
            ],
        )


# ---------------------------------------------------------------------------
# 路由层（FastAPI 可用时）
# ---------------------------------------------------------------------------

try:
    from fastapi import HTTPException as _HTTPException

    from app.main import release_batch_gate
    from app.schemas import ReleaseBatchGateRequest

    _HAS_FASTAPI = True
except ImportError:  # pragma: no cover - CI 中 fastapi 必装
    _HAS_FASTAPI = False


@unittest.skipUnless(_HAS_FASTAPI, "fastapi 未安装")
class ReleaseBatchGateEndpointTests(unittest.TestCase):
    def _request(self, payload):
        return release_batch_gate(
            ReleaseBatchGateRequest.model_validate(payload)
        )

    def test_endpoint_full_response(self):
        resp = self._request({
            "zones": [
                {"name": "example.com", "target": 2024092102,
                 "replicas": [
                     {"name": "ns-a", "current": 2024092101},
                     {"name": "ns-b", "current": 2024092100}]},
                {"name": "api.example.com", "target": 105,
                 "depends_on": ["example.com"],
                 "replicas": [
                     {"name": "ns-c", "current": 100},
                     {"name": "ns-d", "current": 104}]},
                {"name": "static.example.com", "target": 8,
                 "replicas": [{"name": "ns-e", "current": 8}]},
            ],
        })
        self.assertTrue(resp.releasable)
        self.assertEqual(resp.zone_count, 3)
        self.assertEqual(
            [(b.index, b.zones) for b in resp.batches],
            [(0, ["example.com", "static.example.com"]),
             (1, ["api.example.com"])],
        )
        first = resp.zones[0]
        self.assertEqual(first.status, "ready")
        self.assertEqual(first.batch, 0)
        self.assertEqual(first.max_forward_distance, 2)
        self.assertEqual(first.reasons, [])
        self.assertEqual(
            [(r.name, r.relation, r.forward_distance, r.blocking)
             for r in first.replicas],
            [("ns-a", "newer", 1, False), ("ns-b", "newer", 2, False)],
        )
        dependent = resp.zones[1]
        self.assertEqual(dependent.batch, 1)
        self.assertEqual(dependent.max_forward_distance, 5)
        uptodate = resp.zones[2]
        self.assertEqual(uptodate.status, "ready")
        self.assertEqual(uptodate.max_forward_distance, 0)

    def test_endpoint_blocked_response(self):
        resp = self._request({
            "zones": [
                {"name": "a", "target": 10,
                 "replicas": [{"name": "ns-a", "current": 20}]},
                {"name": "b", "target": 10, "depends_on": ["a"],
                 "replicas": [{"name": "ns-b", "current": 1}]},
            ],
        })
        self.assertFalse(resp.releasable)
        self.assertEqual(resp.batches, [])
        self.assertEqual(resp.zones[0].status, "blocked")
        self.assertEqual(resp.zones[0].reasons, ["target_not_newer"])
        self.assertEqual(resp.zones[0].blocking_replicas, ["ns-a"])
        self.assertIsNone(resp.zones[0].batch)
        self.assertEqual(resp.zones[1].reasons, ["dependency_blocked"])
        self.assertEqual(resp.zones[1].blocking_dependencies, ["a"])

    def test_endpoint_wraparound_release(self):
        resp = self._request({
            "zones": [
                {"name": "wrap.example", "target": 2,
                 "replicas": [
                     {"name": "ns-a", "current": 4294967294},
                     {"name": "ns-b", "current": 4294967295}]},
            ],
        })
        self.assertTrue(resp.releasable)
        self.assertEqual(resp.zones[0].max_forward_distance, 4)
        self.assertEqual(
            [r.relation for r in resp.zones[0].replicas], ["newer", "newer"]
        )

    def test_endpoint_duplicate_zone_422(self):
        with self.assertRaises(_HTTPException) as ctx:
            self._request({
                "zones": [
                    {"name": "a", "target": 1,
                     "replicas": [{"name": "ns-a", "current": 0}]},
                    {"name": "a", "target": 2,
                     "replicas": [{"name": "ns-b", "current": 0}]},
                ],
            })
        self.assertEqual(ctx.exception.status_code, 422)
        self.assertEqual(ctx.exception.detail["error"], "duplicate_zone_name")

    def test_endpoint_duplicate_replica_422(self):
        with self.assertRaises(_HTTPException) as ctx:
            self._request({
                "zones": [
                    {"name": "a", "target": 1,
                     "replicas": [
                         {"name": "ns-a", "current": 0},
                         {"name": "ns-a", "current": 1}]},
                ],
            })
        self.assertEqual(ctx.exception.status_code, 422)
        self.assertEqual(
            ctx.exception.detail["error"], "duplicate_replica_name"
        )

    def test_endpoint_unknown_dependency_422(self):
        with self.assertRaises(_HTTPException) as ctx:
            self._request({
                "zones": [
                    {"name": "a", "target": 1, "depends_on": ["ghost"],
                     "replicas": [{"name": "ns-a", "current": 0}]},
                ],
            })
        self.assertEqual(ctx.exception.status_code, 422)
        self.assertEqual(ctx.exception.detail["error"], "unknown_dependency")
        self.assertEqual(ctx.exception.detail["dependency"], "ghost")

    def test_endpoint_dependency_cycle_422(self):
        with self.assertRaises(_HTTPException) as ctx:
            self._request({
                "zones": [
                    {"name": "a", "target": 1, "depends_on": ["b"],
                     "replicas": [{"name": "ns-a", "current": 0}]},
                    {"name": "b", "target": 1, "depends_on": ["a"],
                     "replicas": [{"name": "ns-b", "current": 0}]},
                ],
            })
        self.assertEqual(ctx.exception.status_code, 422)
        detail = ctx.exception.detail
        self.assertEqual(detail["error"], "dependency_cycle")
        self.assertEqual(detail["cycle"], ["a", "b"])

    def test_endpoint_serial_incomparable_422(self):
        with self.assertRaises(_HTTPException) as ctx:
            self._request({
                "zones": [
                    {"name": "a", "target": 2147483648,
                     "replicas": [{"name": "ns-a", "current": 0}]},
                ],
            })
        self.assertEqual(ctx.exception.status_code, 422)
        detail = ctx.exception.detail
        self.assertEqual(detail["error"], "serial_incomparable")
        self.assertEqual(detail["zone"], "a")
        self.assertEqual(detail["replica"], "ns-a")
        self.assertEqual(detail["forward_distance"], 2147483648)

    def test_endpoint_empty_zones_422(self):
        with self.assertRaises(Exception):
            ReleaseBatchGateRequest.model_validate({"zones": []})

    def test_endpoint_duplicate_depends_on_422(self):
        with self.assertRaises(Exception):
            ReleaseBatchGateRequest.model_validate({
                "zones": [
                    {"name": "a", "target": 1,
                     "depends_on": ["b", "b"],
                     "replicas": [{"name": "ns-a", "current": 0}]},
                    {"name": "b", "target": 1,
                     "replicas": [{"name": "ns-b", "current": 0}]},
                ],
            })

    def test_endpoint_blank_zone_name_422(self):
        with self.assertRaises(Exception):
            ReleaseBatchGateRequest.model_validate({
                "zones": [
                    {"name": "   ", "target": 1,
                     "replicas": [{"name": "ns-a", "current": 0}]},
                ],
            })

    def test_endpoint_invalid_target_422(self):
        for bad_target in (MAX_SERIAL + 1, -1, True, 1.5, "10"):
            with self.subTest(bad_target=bad_target):
                with self.assertRaises(Exception):
                    ReleaseBatchGateRequest.model_validate({
                        "zones": [
                            {"name": "a", "target": bad_target,
                             "replicas": [{"name": "ns-a", "current": 0}]},
                        ],
                    })


if __name__ == "__main__":
    unittest.main()
