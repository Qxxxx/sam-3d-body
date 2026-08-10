import assert from "node:assert/strict";
import test from "node:test";

import {
  computeFirstFrameFootSupport,
  computeFrameFootCenters,
} from "./ground-calibration.js";

test("separates two feet and returns their sole midpoint", () => {
  const positions = new Float32Array([
    -0.6, 1.1, 0.0, -0.4, 1.1, 0.0, -0.5, 1.08, 0.2, -0.5, 0.4, 0.1, 0.5, 0.82,
    -0.1, 0.7, 0.82, -0.1, 0.6, 0.8, 0.1, 0.6, 0.3, 0.0,
  ]);
  const indices = new Uint32Array([0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7]);

  const support = computeFirstFrameFootSupport(positions, indices, 8);

  assert.equal(support.footCenters.length, 2);
  assert.deepEqual(
    support.footCenters.map((center) =>
      center.map((value) => +value.toFixed(3)),
    ),
    [
      [-0.5, 1.093, 0.067],
      [0.6, 0.813, -0.033],
    ],
  );
  assert.deepEqual(
    support.midpoint.map((value) => +value.toFixed(3)),
    [0.05, 0.953, 0.017],
  );
  assert.deepEqual(support.footVertexIndices, [
    [0, 1, 2],
    [4, 5, 6],
  ]);
});

test("tracks the calibrated sole vertices while the hips squat", () => {
  const positions = new Float32Array([
    -0.6, 1.1, 0.0, -0.4, 1.1, 0.0, 0.5, 0.9, 0.0, 0.7, 0.9, 0.0, -0.6, 0.7,
    0.1, -0.4, 0.7, 0.1, 0.5, 0.5, -0.1, 0.7, 0.5, -0.1,
  ]);
  const centers = computeFrameFootCenters(positions, 4, 1, [
    [0, 1],
    [2, 3],
  ]);

  assert.deepEqual(
    centers.map((center) => center.map((value) => +value.toFixed(3))),
    [
      [-0.5, 0.7, 0.1],
      [0.6, 0.5, -0.1],
    ],
  );
});

test("rejects an out-of-range support frame", () => {
  assert.throws(
    () => computeFrameFootCenters(new Float32Array(6), 2, 1, [[0], [1]]),
    /超出网格数据范围/,
  );
});

test("rejects a mesh without two lower-body components", () => {
  const positions = new Float32Array([0, 1, 0, 0.1, 1, 0, 0, 0.9, 0.1]);
  const indices = new Uint32Array([0, 1, 2]);

  assert.throws(
    () => computeFirstFrameFootSupport(positions, indices, 3),
    /无法分离左右脚/,
  );
});
