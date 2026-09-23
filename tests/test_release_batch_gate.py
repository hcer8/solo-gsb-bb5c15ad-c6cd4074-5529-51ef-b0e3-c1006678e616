"""POST /release-batch-gate 多 zone 分批发布门禁单元测试。

运行：python -m unittest discover -s tests -v
"""

from __future__ import annotations

import copy
import unittest

from app.serial import (
    HALF,
    MAX_SERIAL,
    ReleaseBatchValidationError,
    evaluate_release_batch_gate,
    validate_release_batch_input,
)


def zone(name, currents, target, requires=None):
    return (name, list(currents), target, list(requires or []))


class ValidateReleaseBatchInputTests(unittest.TestCase):
    def test_valid_input_passes(self):
        validate_release_batch_input(
            [
                zone("a", [1, 2], 3),
                zone("b", [2], 4, ["a"]),
                zone("c", [MAX_SERIAL], 0, ["a", "b"]),
            ]
        )

    def test_older_serial_is_not_a_validation_error(self):
        # 回退副本是 200 响应中的阻塞原因，不是 422 结构化错误
        validate_release_batch_input([zone("a", [300], 250)])

    def test_duplicate_zone(self):
        with self.assertRaises(ReleaseBatchValidationError) as ctx:
            validate_release_batch_input(
                [zone("a", [1], 2), zone("b", [1], 2), zone("a", [1], 2)]
            )
        detail = ctx.exception.detail
        self.assertEqual(detail["error"], "duplicate_zone")
        self.assertEqual(detail["zone"], "a")
        self.assertEqual(detail["first_index"], 0)
        self.assertEqual(detail["index"], 2)

    def test_unknown_dependency(self):
        with self.assertRaises(ReleaseBatchValidationError) as ctx:
            validate_release_batch_input(
                [zone("a", [1], 2), zone("b", [1], 2, ["a", "x"])]
            )
        detail = ctx.exception.detail
        self.assertEqual(detail["error"], "unknown_dependency")
        self.assertEqual(detail["zone"], "b")
        self.assertEqual(detail["dependency"], "x")

    def test_self_dependency(self):
        with self.assertRaises(ReleaseBatchValidationError) as ctx:
            validate_release_batch_input([zone("a", [1], 2, ["a"])])
        detail = ctx.exception.detail
        self.assertEqual(detail["error"], "self_dependency")
        self.assertEqual(detail["zone"], "a")

    def test_duplicate_dependency(self):
        with self.assertRaises(ReleaseBatchValidationError) as ctx:
            validate_release_batch_input(
                [zone("a", [1], 2), zone("b", [1], 2, ["a", "a"])]
            )
        detail = ctx.exception.detail
        self.assertEqual(detail["error"], "duplicate_dependency")
        self.assertEqual(detail["zone"], "b")
        self.assertEqual(detail["dependency"], "a")

    def test_two_node_cycle(self):
        with self.assertRaises(ReleaseBatchValidationError) as ctx:
            validate_release_batch_input(
                [zone("a", [1], 2, ["b"]), zone("b", [1], 2, ["a"])]
            )
        detail = ctx.exception.detail
        self.assertEqual(detail["error"], "dependency_cycle")
        self.assertEqual(detail["cycle"], ["a", "b", "a"])
        self.assertEqual(
            detail["cycle_edges"],
            [
                {"zone": "a", "requires": "b"},
                {"zone": "b", "requires": "a"},
            ],
        )

    def test_three_node_cycle_normalized_from_smallest_zone(self):
        # 请求顺序与环的最小起点不同：环仍规范化为从最小 zone 名起
        with self.assertRaises(ReleaseBatchValidationError) as ctx:
            validate_release_batch_input(
                [
                    zone("zzz", [1], 2, ["mid"]),
                    zone("mid", [1], 2, ["aaa"]),
                    zone("aaa", [1], 2, ["zzz"]),
                ]
            )
        detail = ctx.exception.detail
        self.assertEqual(detail["error"], "dependency_cycle")
        # 行进方向 aaa -> zzz -> mid -> aaa
        self.assertEqual(detail["cycle"], ["aaa", "zzz", "mid", "aaa"])

    def test_cycle_entered_from_unrelated_root(self):
        with self.assertRaises(ReleaseBatchValidationError) as ctx:
            validate_release_batch_input(
                [
                    zone("root", [1], 2, ["b"]),
                    zone("b", [1], 2, ["c"]),
                    zone("c", [1], 2, ["b"]),
                ]
            )
        self.assertEqual(ctx.exception.detail["cycle"], ["b", "c", "b"])

    def test_antipodal_pair_incomparable(self):
        with self.assertRaises(ReleaseBatchValidationError) as ctx:
            validate_release_batch_input(
                [zone("a", [5, 0], HALF)]
            )
        detail = ctx.exception.detail
        self.assertEqual(detail["error"], "incomparable_serials")
        self.assertEqual(detail["zone"], "a")
        self.assertEqual(detail["replica_index"], 1)
        self.assertEqual(detail["current"], 0)
        self.assertEqual(detail["target"], HALF)
        self.assertEqual(detail["forward_distance"], HALF)

    def test_incomparable_first_zone_in_request_order(self):
        with self.assertRaises(ReleaseBatchValidationError) as ctx:
            validate_release_batch_input(
                [
                    zone("ok", [1], 2),
                    zone("bad", [10], (10 + HALF) % (2**32)),
                ]
            )
        self.assertEqual(ctx.exception.detail["zone"], "bad")

    def test_validation_order_structure_before_serials(self):
        # 同时存在依赖环与正对点：先报结构错误（依赖环）
        with self.assertRaises(ReleaseBatchValidationError) as ctx:
            validate_release_batch_input(
                [
                    zone("a", [0], HALF, ["b"]),
                    zone("b", [0], HALF, ["a"]),
                ]
            )
        self.assertEqual(ctx.exception.detail["error"], "dependency_cycle")


class EvaluateReleaseBatchGateTests(unittest.TestCase):
    def test_single_zone_all_newer(self):
        outcome = evaluate_release_batch_gate([zone("a", [100, 98], 110)])
        self.assertTrue(outcome["scheduled"])
        self.assertEqual(outcome["batch_count"], 1)
        self.assertEqual(outcome["max_forward_distance"], 12)
        self.assertEqual(outcome["batches"], [{"batch": 1, "zones": ["a"]}])
        self.assertEqual(outcome["blocked_zones"], [])

        item = outcome["zones"][0]
        self.assertEqual(item["zone"], "a")
        self.assertEqual(item["status"], "scheduled")
        self.assertEqual(item["batch"], 1)
        self.assertTrue(item["serial_ready"])
        self.assertIsNone(item["blocking_reason"])
        self.assertEqual(item["blocked_by"], [])
        self.assertEqual(
            [(r["index"], r["relation"], r["forward_distance"], r["blocking"])
             for r in item["replicas"]],
            [(0, "newer", 10, False), (1, "newer", 12, False)],
        )
        for replica in item["replicas"]:
            self.assertEqual(replica["current"] + 0, replica["current"])
            self.assertEqual(replica["target"], 110)

    def test_wrap_around_has_small_forward_distances(self):
        outcome = evaluate_release_batch_gate(
            [zone("a", [MAX_SERIAL - 5, MAX_SERIAL - 1], 0)]
        )
        self.assertTrue(outcome["scheduled"])
        self.assertEqual(outcome["max_forward_distance"], 6)
        relations = [r["relation"] for r in outcome["zones"][0]["replicas"]]
        self.assertEqual(relations, ["newer", "newer"])

    def test_equal_replica_distance_zero_ready(self):
        outcome = evaluate_release_batch_gate([zone("a", [5, 4], 5)])
        self.assertTrue(outcome["scheduled"])
        self.assertEqual(outcome["max_forward_distance"], 1)
        self.assertEqual(
            [(r["relation"], r["forward_distance"], r["blocking"])
             for r in outcome["zones"][0]["replicas"]],
            [("equal", 0, False), ("newer", 1, False)],
        )

    def test_older_replica_blocks_zone(self):
        outcome = evaluate_release_batch_gate([zone("a", [300, 200], 250)])
        self.assertFalse(outcome["scheduled"])
        self.assertEqual(outcome["batch_count"], 0)
        self.assertEqual(outcome["max_forward_distance"], 0)
        self.assertEqual(outcome["batches"], [])
        self.assertEqual(outcome["blocked_zones"], ["a"])

        item = outcome["zones"][0]
        self.assertEqual(item["status"], "blocked")
        self.assertIsNone(item["batch"])
        self.assertFalse(item["serial_ready"])
        self.assertEqual(item["blocking_reason"], "older_serial")
        self.assertEqual(item["blocked_by"], [])
        # 最大前向距离仍按全部副本计算（含回退副本的巨大前向距离）
        self.assertEqual(
            item["max_forward_distance"], (250 - 300) % (2**32)
        )
        self.assertEqual(
            [(r["index"], r["relation"], r["blocking"])
             for r in item["replicas"]],
            [(0, "older", True), (1, "newer", False)],
        )

    def test_linear_chain_batches(self):
        outcome = evaluate_release_batch_gate(
            [
                zone("a", [1], 2),
                zone("b", [1], 2, ["a"]),
                zone("c", [1], 2, ["b"]),
            ]
        )
        self.assertTrue(outcome["scheduled"])
        self.assertEqual(outcome["batch_count"], 3)
        self.assertEqual(
            outcome["batches"],
            [
                {"batch": 1, "zones": ["a"]},
                {"batch": 2, "zones": ["b"]},
                {"batch": 3, "zones": ["c"]},
            ],
        )
        self.assertEqual([z["batch"] for z in outcome["zones"]], [1, 2, 3])

    def test_diamond_dag_levels_and_sorting(self):
        outcome = evaluate_release_batch_gate(
            [
                zone("d", [1], 2, ["b", "c"]),
                zone("c", [1], 2, ["a"]),
                zone("b", [1], 2, ["a"]),
                zone("a", [1], 2),
            ]
        )
        self.assertTrue(outcome["scheduled"])
        # 批次按 zone 名升序；zones 保持请求顺序 d,c,b,a
        self.assertEqual(
            outcome["batches"],
            [
                {"batch": 1, "zones": ["a"]},
                {"batch": 2, "zones": ["b", "c"]},
                {"batch": 3, "zones": ["d"]},
            ],
        )
        by_name = {z["zone"]: z["batch"] for z in outcome["zones"]}
        self.assertEqual(by_name, {"a": 1, "b": 2, "c": 2, "d": 3})

    def test_level_uses_longest_prerequisite_path(self):
        # d 同时依赖第 1 批的 a 与第 2 批的 b -> 必须在第 3 批
        outcome = evaluate_release_batch_gate(
            [
                zone("a", [1], 2),
                zone("b", [1], 2, ["a"]),
                zone("d", [1], 2, ["a", "b"]),
            ]
        )
        by_name = {z["zone"]: z["batch"] for z in outcome["zones"]}
        self.assertEqual(by_name, {"a": 1, "b": 2, "d": 3})

    def test_blocked_root_cascades_through_chain(self):
        outcome = evaluate_release_batch_gate(
            [
                zone("a", [300], 250),
                zone("b", [40], 50, ["a"]),
                zone("c", [1], 2, ["b"]),
            ]
        )
        self.assertFalse(outcome["scheduled"])
        self.assertEqual(outcome["batches"], [])
        self.assertEqual(outcome["blocked_zones"], ["a", "b", "c"])
        a, b, c = outcome["zones"]
        self.assertEqual(a["blocking_reason"], "older_serial")
        self.assertEqual(b["blocking_reason"], "dependency_unsatisfied")
        self.assertEqual(b["blocked_by"], ["a"])
        self.assertEqual(c["blocking_reason"], "dependency_unsatisfied")
        self.assertEqual(c["blocked_by"], ["b"])

    def test_partial_graph_keeps_schedulable_branch(self):
        # a 可发布（第 1 批）；b 自身回退阻塞；c 仅依赖 b -> blocked；
        # d 依赖 a 与 b -> blocked，blocked_by 只列未就位的 b
        outcome = evaluate_release_batch_gate(
            [
                zone("a", [1], 2),
                zone("b", [9], 5),
                zone("c", [1], 2, ["b"]),
                zone("d", [1], 2, ["a", "b"]),
            ]
        )
        self.assertFalse(outcome["scheduled"])
        self.assertEqual(outcome["batches"], [{"batch": 1, "zones": ["a"]}])
        self.assertEqual(outcome["batch_count"], 1)
        # 顶层最大距离只统计已编入批次的 zone
        self.assertEqual(outcome["max_forward_distance"], 1)
        by_name = {z["zone"]: z for z in outcome["zones"]}
        self.assertEqual(by_name["a"]["status"], "scheduled")
        self.assertEqual(by_name["b"]["blocking_reason"], "older_serial")
        self.assertEqual(by_name["b"]["blocked_by"], [])
        self.assertEqual(by_name["c"]["blocking_reason"],
                         "dependency_unsatisfied")
        self.assertEqual(by_name["c"]["blocked_by"], ["b"])
        self.assertEqual(by_name["d"]["blocking_reason"],
                         "dependency_unsatisfied")
        self.assertEqual(by_name["d"]["blocked_by"], ["b"])
        self.assertEqual(outcome["blocked_zones"], ["b", "c", "d"])

    def test_older_serial_takes_precedence_over_missing_prerequisite(self):
        # 前置未就位且自身也有回退副本：原因是 older_serial
        outcome = evaluate_release_batch_gate(
            [
                zone("a", [9], 5),
                zone("b", [99], 50, ["a"]),
            ]
        )
        b = outcome["zones"][1]
        self.assertEqual(b["status"], "blocked")
        self.assertEqual(b["blocking_reason"], "older_serial")
        self.assertTrue(b["serial_ready"] is False)

    def test_blocked_by_sorted_and_independent_branches(self):
        outcome = evaluate_release_batch_gate(
            [
                zone("a", [9], 5),
                zone("z", [9], 5),
                zone("m", [1], 2, ["z", "a"]),
            ]
        )
        m = outcome["zones"][2]
        self.assertEqual(m["blocking_reason"], "dependency_unsatisfied")
        self.assertEqual(m["blocked_by"], ["a", "z"])

    def test_requires_order_does_not_affect_result(self):
        payload = [
            zone("a", [1], 2),
            zone("b", [1], 2, ["a"]),
            zone("c", [1], 2),
            zone("d", [1], 2, ["c", "b"]),
        ]
        shuffled = [
            zone("a", [1], 2),
            zone("b", [1], 2, ["a"]),
            zone("c", [1], 2),
            zone("d", [1], 2, ["b", "c"]),
        ]
        self.assertEqual(
            evaluate_release_batch_gate(payload),
            evaluate_release_batch_gate(shuffled),
        )

    def test_deterministic_repeated_evaluation(self):
        payload = [
            zone("a", [100, 98], 110, []),
            zone("b", [105], 120, ["a"]),
            zone("c", [MAX_SERIAL - 5], 0, ["a"]),
        ]
        first = evaluate_release_batch_gate(copy.deepcopy(payload))
        second = evaluate_release_batch_gate(copy.deepcopy(payload))
        self.assertEqual(first, second)

    def test_deep_chain_does_not_use_recursion(self):
        # 远超默认递归深度的线性链也必须正常分层（迭代 Kahn / 迭代 DFS）
        n = 12000
        zones_ = [zone(f"z{i:06d}", [1], 2, [f"z{i-1:06d}"] if i else [])
                  for i in range(n)]
        validate_release_batch_input(zones_)
        outcome = evaluate_release_batch_gate(zones_)
        self.assertTrue(outcome["scheduled"])
        self.assertEqual(outcome["batch_count"], n)
        self.assertEqual(outcome["batches"][0]["zones"], ["z000000"])
        self.assertEqual(outcome["batches"][-1]["zones"],
                         [f"z{n - 1:06d}"])

    def test_top_max_distance_ignores_blocked_zones(self):
        outcome = evaluate_release_batch_gate(
            [
                zone("a", [1], 10),            # 距离 9，编入
                zone("b", [100], 50, ["a"]),   # (50-100) mod 2^32 > 2^31 -> older，阻塞
            ]
        )
        # b 相对目标为 older（回退）-> blocked，其距离不计入顶层最大值
        self.assertEqual(outcome["zones"][1]["status"], "blocked")
        self.assertEqual(outcome["max_forward_distance"], 9)


# 路由层测试：直接以函数方式调用 FastAPI 端点（无需 httpx/TestClient）。
try:
    from fastapi import HTTPException as _HTTPException
    from pydantic import ValidationError as _ValidationError

    from app.main import app, release_batch_gate
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

    def _expect_422(self, payload, error):
        with self.assertRaises(_HTTPException) as ctx:
            self._request(payload)
        self.assertEqual(ctx.exception.status_code, 422)
        self.assertEqual(ctx.exception.detail["error"], error)
        return ctx.exception.detail

    def test_endpoint_happy_path(self):
        resp = self._request(
            {
                "zones": [
                    {
                        "zone": "example.com",
                        "replicas": [
                            {"name": "ns1", "current": 100},
                            {"name": "ns2", "current": 98},
                        ],
                        "target": 110,
                        "requires": [],
                    },
                    {
                        "zone": "cdn.example.com",
                        "replicas": [{"name": "ns1", "current": 105}],
                        "target": 120,
                        "requires": ["example.com"],
                    },
                    {
                        "zone": "api.example.com",
                        "replicas": [
                            {"name": "ns1", "current": MAX_SERIAL - 5},
                            {"name": "ns2", "current": MAX_SERIAL - 1},
                        ],
                        "target": 0,
                        "requires": ["example.com"],
                    },
                ]
            }
        )
        self.assertTrue(resp.scheduled)
        self.assertEqual(resp.batch_count, 2)
        self.assertEqual(resp.max_forward_distance, 15)
        self.assertEqual(
            [b.model_dump() for b in resp.batches],
            [
                {"batch": 1, "zones": ["example.com"]},
                {"batch": 2, "zones": ["api.example.com", "cdn.example.com"]},
            ],
        )
        self.assertEqual([z.zone for z in resp.zones],
                         ["example.com", "cdn.example.com", "api.example.com"])
        replica = resp.zones[0].replicas[0]
        self.assertEqual(replica.name, "ns1")
        self.assertEqual(replica.index, 0)
        self.assertEqual(replica.relation, "newer")
        self.assertEqual(replica.forward_distance, 10)
        self.assertFalse(replica.blocking)

    def test_endpoint_blocked_response(self):
        resp = self._request(
            {
                "zones": [
                    {
                        "zone": "blocked.example",
                        "replicas": [
                            {"name": "ns1", "current": 300},
                            {"name": "ns2", "current": 200},
                        ],
                        "target": 250,
                        "requires": [],
                    },
                    {
                        "zone": "waiting.example",
                        "replicas": [
                            {"name": "ns1", "current": 40},
                            {"name": "ns2", "current": 41},
                        ],
                        "target": 50,
                        "requires": ["blocked.example"],
                    },
                ]
            }
        )
        self.assertFalse(resp.scheduled)
        self.assertEqual(resp.batches, [])
        self.assertEqual(resp.batch_count, 0)
        self.assertEqual(resp.max_forward_distance, 0)
        self.assertEqual(resp.blocked_zones,
                         ["blocked.example", "waiting.example"])
        blocked, waiting = resp.zones
        self.assertEqual(blocked.status, "blocked")
        self.assertIsNone(blocked.batch)
        self.assertEqual(blocked.blocking_reason, "older_serial")
        self.assertTrue(blocked.replicas[0].blocking)
        self.assertEqual(waiting.blocking_reason, "dependency_unsatisfied")
        self.assertEqual(waiting.blocked_by, ["blocked.example"])

    def test_duplicate_zone_422(self):
        detail = self._expect_422(
            {
                "zones": [
                    {"zone": "a", "replicas": [{"name": "r", "current": 1}],
                     "target": 2},
                    {"zone": "a", "replicas": [{"name": "r", "current": 1}],
                     "target": 2},
                ]
            },
            "duplicate_zone",
        )
        self.assertEqual(detail["first_index"], 0)
        self.assertEqual(detail["index"], 1)

    def test_unknown_dependency_422(self):
        detail = self._expect_422(
            {
                "zones": [
                    {"zone": "a", "replicas": [{"name": "r", "current": 1}],
                     "target": 2, "requires": ["ghost"]},
                ]
            },
            "unknown_dependency",
        )
        self.assertEqual(detail["zone"], "a")
        self.assertEqual(detail["dependency"], "ghost")

    def test_self_dependency_422(self):
        self._expect_422(
            {
                "zones": [
                    {"zone": "a", "replicas": [{"name": "r", "current": 1}],
                     "target": 2, "requires": ["a"]},
                ]
            },
            "self_dependency",
        )

    def test_duplicate_dependency_422(self):
        detail = self._expect_422(
            {
                "zones": [
                    {"zone": "a", "replicas": [{"name": "r", "current": 1}],
                     "target": 2},
                    {"zone": "b", "replicas": [{"name": "r", "current": 1}],
                     "target": 2, "requires": ["a", "a"]},
                ]
            },
            "duplicate_dependency",
        )
        self.assertEqual(detail["zone"], "b")

    def test_dependency_cycle_422(self):
        detail = self._expect_422(
            {
                "zones": [
                    {"zone": "a", "replicas": [{"name": "r", "current": 1}],
                     "target": 3, "requires": ["b"]},
                    {"zone": "b", "replicas": [{"name": "r", "current": 1}],
                     "target": 3, "requires": ["a"]},
                ]
            },
            "dependency_cycle",
        )
        self.assertEqual(detail["cycle"], ["a", "b", "a"])
        self.assertEqual(len(detail["cycle_edges"]), 2)

    def test_incomparable_serials_422(self):
        detail = self._expect_422(
            {
                "zones": [
                    {"zone": "a",
                     "replicas": [
                         {"name": "r1", "current": 0},
                         {"name": "r2", "current": 1},
                     ],
                     "target": HALF},
                ]
            },
            "incomparable_serials",
        )
        self.assertEqual(detail["zone"], "a")
        self.assertEqual(detail["replica_index"], 0)
        self.assertEqual(detail["forward_distance"], HALF)

    def test_duplicate_replica_name_inside_zone_422(self):
        detail = self._expect_422(
            {
                "zones": [
                    {"zone": "a",
                     "replicas": [
                         {"name": "r", "current": 1},
                         {"name": "r", "current": 2},
                     ],
                     "target": 3},
                ]
            },
            "duplicate_replica_name",
        )
        self.assertEqual(detail["zone"], "a")
        self.assertEqual(detail["name"], "r")

    def test_replica_names_may_repeat_across_zones(self):
        resp = self._request(
            {
                "zones": [
                    {"zone": "a", "replicas": [{"name": "r", "current": 1}],
                     "target": 2},
                    {"zone": "b", "replicas": [{"name": "r", "current": 1}],
                     "target": 2},
                ]
            }
        )
        self.assertTrue(resp.scheduled)

    def test_schema_validation_errors(self):
        base = {
            "zones": [
                {"zone": "a", "replicas": [{"name": "r", "current": 1}],
                 "target": 2},
            ]
        }

        def mutated(**changes):
            payload = copy.deepcopy(base)
            for key, value in changes.items():
                payload[key] = value
            return payload

        invalid_payloads = [
            mutated(zones=[]),
            {"zones": [{"zone": "a", "replicas": [], "target": 2}]},
            {"zones": [{"zone": "  ",
                        "replicas": [{"name": "r", "current": 1}],
                        "target": 2}]},
            {"zones": [{"zone": "a",
                        "replicas": [{"name": "r", "current": 2**32}],
                        "target": 2}]},
            {"zones": [{"zone": "a",
                        "replicas": [{"name": "r", "current": -1}],
                        "target": 2}]},
            {"zones": [{"zone": "a",
                        "replicas": [{"name": "r", "current": True}],
                        "target": 2}]},
            {"zones": [{"zone": "a",
                        "replicas": [{"name": "r", "current": 1}],
                        "target": 2**32}]},
            {"zones": [{"zone": "a",
                        "replicas": [{"name": "r", "current": 1}],
                        "target": "2"}]},
            {"zones": [{"zone": "a",
                        "replicas": [{"name": "r", "current": 1}],
                        "target": 2, "requires": [""]}]},
            {"zones": [{"zone": "a",
                        "replicas": [{"name": "r", "current": 1}],
                        "target": 2, "requires": ["  "]}]},
            {"zones": [{"zone": "a",
                        "replicas": [{"name": "r", "current": 1}],
                        "target": 2, "requires": [1]}]},
            {"zones": [{"zone": "a",
                        "replicas": [{"name": "r", "current": 1}],
                        "target": 2, "requires": "a"}]},
            {"zones": [{"zone": "a",
                        "replicas": [{"name": "  ", "current": 1}],
                        "target": 2}]},
        ]
        for payload in invalid_payloads:
            with self.assertRaises(
                _ValidationError, msg=f"payload={payload}"
            ):
                ReleaseBatchGateRequest.model_validate(payload)

    def test_requires_defaults_to_empty_list(self):
        resp = self._request(
            {
                "zones": [
                    {"zone": "a",
                     "replicas": [{"name": "r", "current": 1}],
                     "target": 2},
                ]
            }
        )
        self.assertEqual(resp.batches[0].zones, ["a"])

    def test_openapi_lists_endpoint_with_examples(self):
        schema = app.openapi()
        self.assertIn("/release-batch-gate", schema["paths"])
        self.assertIn("post", schema["paths"]["/release-batch-gate"])
        request_schema = schema["components"]["schemas"]["ReleaseBatchGateRequest"]
        self.assertIn("examples", request_schema)
        self.assertGreaterEqual(len(request_schema["examples"]), 2)
        # 示例本身必须能通过请求模型校验并被端点处理
        for example in request_schema["examples"]:
            request = ReleaseBatchGateRequest.model_validate(example)
            release_batch_gate(request)


if __name__ == "__main__":
    unittest.main()
