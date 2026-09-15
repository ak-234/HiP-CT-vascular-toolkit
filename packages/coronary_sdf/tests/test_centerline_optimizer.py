"""Acceptance tests for constrained radius-normalized centreline smoothing."""
from __future__ import annotations

import unittest

import numpy as np
from scipy.spatial import cKDTree

from coronary_sdf.centerline_optimizer import (
    smooth_centerlines_constrained_multiscale,
)


def _single(coords_mm: np.ndarray, radius_mm: float = 0.5):
    points = {
        i: (*tuple(np.asarray(p) * 1000.0), radius_mm * 1000.0)
        for i, p in enumerate(coords_mm)
    }
    segments = [dict(id=0, node1=0, node2=1, point_ids=list(points))]
    nodes = {
        0: (*tuple(coords_mm[0] * 1000.0), 1),
        1: (*tuple(coords_mm[-1] * 1000.0), 1),
    }
    return nodes, points, segments


def _coordinates(points):
    return np.asarray([points[i][:3] for i in sorted(points)], dtype=float) / 1000.0


def _resample(polyline: np.ndarray, count: int = 1001) -> np.ndarray:
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(polyline, axis=0), axis=1))]
    target = np.linspace(0.0, arc[-1], count)
    return np.column_stack([np.interp(target, arc, polyline[:, k]) for k in range(3)])


class ConstrainedCenterlineTests(unittest.TestCase):
    def test_noisy_line_is_smoothed_with_fixed_endpoints_radii_and_drift(self):
        x = np.linspace(0.0, 10.0, 31)
        y = np.zeros_like(x)
        y[1:-1] = 0.08 * np.sin(np.arange(1, len(x)-1) * 2.3)
        raw = np.column_stack((x, y, np.zeros_like(x)))
        nodes, points, segments = _single(raw)
        original_radii = [points[i][3] for i in points]
        output, report = smooth_centerlines_constrained_multiscale(nodes, points, segments)
        fitted = _coordinates(output)
        np.testing.assert_array_equal(fitted[[0, -1]], raw[[0, -1]])
        self.assertEqual([output[i][3] for i in output], original_radii)
        self.assertLessEqual(report.max_displacement_radius, 0.25 + 1e-12)
        self.assertEqual(report.curvature_violations_after, 0)
        self.assertLess(np.std(fitted[1:-1, 1]), np.std(raw[1:-1, 1]))

    def test_global_scale_equivariance(self):
        x = np.linspace(0.0, 10.0, 31)
        raw = np.column_stack((x, 0.08*np.sin(np.arange(31)*2.3), np.zeros(31)))
        normalized = []
        for scale in (0.01, 1.0, 100.0):
            nodes, points, segments = _single(raw*scale, radius_mm=0.5*scale)
            output, _ = smooth_centerlines_constrained_multiscale(nodes, points, segments)
            normalized.append(_coordinates(output)/scale)
        np.testing.assert_allclose(normalized[0], normalized[1], atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(normalized[1], normalized[2], atol=1e-12, rtol=1e-12)

    def test_sampling_density_invariance(self):
        fitted = []
        for count in (41, 81):
            x = np.linspace(0.0, 10.0, count)
            y = 0.02*np.sin(2*np.pi*x/1.5)*np.sin(np.pi*x/10.0)
            nodes, points, segments = _single(np.column_stack((x,y,np.zeros(count))))
            output, _ = smooth_centerlines_constrained_multiscale(nodes, points, segments)
            fitted.append(_resample(_coordinates(output)))
        distance = max(
            cKDTree(fitted[0]).query(fitted[1])[0].max(),
            cKDTree(fitted[1]).query(fitted[0])[0].max(),
        )
        self.assertLess(distance / 0.5, 0.02)

    def test_separate_parallel_branches_do_not_acquire_an_intersection(self):
        x = np.linspace(0.0, 8.0, 41)
        points = {}
        segments = []
        nodes = {}
        for branch, offset in enumerate((-0.58, 0.58)):
            ids = []
            for k, xv in enumerate(x):
                pid = branch*100+k
                # Noise is antisymmetric and locally narrows the gap.
                y = offset + (1 if branch == 0 else -1)*0.025*np.sin(2.1*k)
                points[pid] = (1000*xv, 1000*y, 0.0, 500.0)
                ids.append(pid)
            nodes[2*branch] = points[ids[0]][:3] + (1,)
            nodes[2*branch+1] = points[ids[-1]][:3] + (1,)
            segments.append(dict(id=branch,node1=2*branch,node2=2*branch+1,point_ids=ids))
        _output, report = smooth_centerlines_constrained_multiscale(nodes,points,segments)
        self.assertEqual(report.input_overlaps, 0)
        self.assertEqual(report.new_branch_conflicts, 0)

    def test_preexisting_overlap_is_frozen_and_reported(self):
        points={};segments=[];nodes={};x=np.linspace(0,4,17)
        for branch,offset in enumerate((-0.4,0.4)):
            ids=[]
            for k,xv in enumerate(x):
                pid=branch*100+k;points[pid]=(1000*xv,1000*offset,0.,500.);ids.append(pid)
            nodes[2*branch]=points[ids[0]][:3]+(1,);nodes[2*branch+1]=points[ids[-1]][:3]+(1,)
            segments.append(dict(id=branch,node1=2*branch,node2=2*branch+1,point_ids=ids))
        output,report=smooth_centerlines_constrained_multiscale(nodes,points,segments)
        self.assertGreater(report.input_overlaps,0)
        self.assertEqual(report.modified_points,0)
        self.assertEqual(output,points)

    def test_infeasible_tight_bend_is_reported_not_raised(self):
        raw=np.asarray([[0.,0.,0.],[1.,0.,0.],[1.,.15,0.],[1.,2.,0.]])
        nodes,points,segments=_single(raw,radius_mm=.8)
        _output,report=smooth_centerlines_constrained_multiscale(nodes,points,segments)
        self.assertGreater(report.unresolved_constraints,0)
        self.assertLessEqual(report.max_displacement_radius,.25+1e-12)

    def test_feasible_curved_tube_reaches_curvature_bound(self):
        angle=np.linspace(0,np.pi/2,41);radial=4.0+.025*np.sin(np.arange(41)*2.4)
        radial[[0,-1]]=4.0
        raw=np.column_stack((radial*np.cos(angle),radial*np.sin(angle),np.zeros(41)))
        nodes,points,segments=_single(raw,radius_mm=.5)
        _output,report=smooth_centerlines_constrained_multiscale(nodes,points,segments)
        self.assertEqual(report.curvature_violations_after,0)
        self.assertEqual(report.new_branch_conflicts,0)

    def test_infeasible_self_overlapping_hairpin_is_preserved_and_reported(self):
        raw=np.asarray([[0,0,0],[1,0,0],[2,0,0],[3,0,0],[4,0,0],
                        [4,.8,0],[3,.8,0],[2,.8,0],[1,.8,0],[0,.8,0]],float)
        nodes,points,segments=_single(raw,radius_mm=.5)
        output,report=smooth_centerlines_constrained_multiscale(nodes,points,segments)
        self.assertGreater(report.self_distance_violations_before,0)
        self.assertGreater(report.unresolved_constraints,0)
        self.assertEqual(output,points)

    def test_y_and_degree_five_junctions_keep_shared_node_fixed(self):
        for degree in (3, 5):
            points={0:(0.,0.,0.,400.)};segments=[];nodes={7:(0.,0.,0.,degree)}
            next_pid=1
            for branch in range(degree):
                angle=2*np.pi*branch/degree
                direction=np.array([np.cos(angle),np.sin(angle),.15*(-1)**branch])
                direction/=np.linalg.norm(direction)
                ids=[0]
                for k in range(1,13):
                    pid=next_pid;next_pid+=1;p=direction*(k/2)
                    if k<12:p=p+.02*np.sin(2.2*k)*np.array([-direction[1],direction[0],0.])
                    points[pid]=(*tuple(p*1000),400.);ids.append(pid)
                segments.append(dict(id=branch,node1=7,node2=100+branch,point_ids=ids))
                nodes[100+branch]=points[ids[-1]][:3]+(1,)
            output,report=smooth_centerlines_constrained_multiscale(nodes,points,segments)
            self.assertEqual(output[0],points[0])
            self.assertEqual(report.new_branch_conflicts,0)
            self.assertLessEqual(report.max_displacement_radius,.25+1e-12)


if __name__ == "__main__":
    unittest.main()
