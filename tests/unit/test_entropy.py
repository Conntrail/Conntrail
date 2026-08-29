"""
Tests for routing_entropy.
"""
import pytest

from conntrail.utils.entropy import routing_entropy


class TestRoutingEntropy:
    def test_all_same_routes(self):
        assert routing_entropy(["a", "a", "a", "a"]) == 0.0

    def test_all_different_routes(self):
        assert routing_entropy(["a", "b", "c", "d"]) == pytest.approx(1.0)

    def test_half_half_routes(self):
        score = routing_entropy(["a", "a", "b", "b"])
        assert 0.4 < score < 0.6

    def test_three_same_one_different(self):
        score = routing_entropy(["a", "a", "a", "b"])
        assert 0.0 < score < 0.5

    def test_empty_list(self):
        assert routing_entropy([]) == 0.0

    def test_single_element(self):
        assert routing_entropy(["a"]) == 0.0

    def test_two_elements_same(self):
        assert routing_entropy(["x", "x"]) == 0.0

    def test_two_elements_different(self):
        assert routing_entropy(["x", "y"]) == pytest.approx(1.0)
