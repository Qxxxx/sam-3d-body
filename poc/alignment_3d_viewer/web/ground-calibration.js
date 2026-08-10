const LOWER_LEG_DEPTH = 0.9;
const SOLE_CONTACT_BAND = 0.04;

function findRoot(parent, index) {
  let root = index;
  while (parent[root] !== root) root = parent[root];
  while (parent[index] !== index) {
    const next = parent[index];
    parent[index] = root;
    index = next;
  }
  return root;
}

function unionRoots(parent, sizes, first, second) {
  let firstRoot = findRoot(parent, first);
  let secondRoot = findRoot(parent, second);
  if (firstRoot === secondRoot) return;
  if (sizes[firstRoot] < sizes[secondRoot]) {
    [firstRoot, secondRoot] = [secondRoot, firstRoot];
  }
  parent[secondRoot] = firstRoot;
  sizes[firstRoot] += sizes[secondRoot];
}

export function computeFirstFrameFootSupport(positions, indices, vertexCount) {
  if (positions.length < vertexCount * 3) {
    throw new Error("第一帧网格数据不完整，无法标定双脚");
  }

  let maximumY = -Infinity;
  for (let vertex = 0; vertex < vertexCount; vertex += 1) {
    maximumY = Math.max(maximumY, positions[vertex * 3 + 1]);
  }
  const lowerLegThreshold = maximumY - LOWER_LEG_DEPTH;
  const parent = new Int32Array(vertexCount);
  const sizes = new Int32Array(vertexCount);
  parent.fill(-1);
  for (let vertex = 0; vertex < vertexCount; vertex += 1) {
    if (positions[vertex * 3 + 1] >= lowerLegThreshold) {
      parent[vertex] = vertex;
      sizes[vertex] = 1;
    }
  }

  for (let offset = 0; offset < indices.length; offset += 3) {
    const a = indices[offset];
    const b = indices[offset + 1];
    const c = indices[offset + 2];
    if (parent[a] >= 0 && parent[b] >= 0) unionRoots(parent, sizes, a, b);
    if (parent[b] >= 0 && parent[c] >= 0) unionRoots(parent, sizes, b, c);
    if (parent[c] >= 0 && parent[a] >= 0) unionRoots(parent, sizes, c, a);
  }

  const componentSizes = new Map();
  for (let vertex = 0; vertex < vertexCount; vertex += 1) {
    if (parent[vertex] < 0) continue;
    const root = findRoot(parent, vertex);
    componentSizes.set(root, (componentSizes.get(root) ?? 0) + 1);
  }
  const footRoots = [...componentSizes.entries()]
    .sort((first, second) => second[1] - first[1])
    .slice(0, 2)
    .map(([root]) => root);
  if (footRoots.length < 2) {
    throw new Error("第一帧无法分离左右脚，不能建立落地标定");
  }

  const componentMaximumY = new Map(footRoots.map((root) => [root, -Infinity]));
  for (let vertex = 0; vertex < vertexCount; vertex += 1) {
    if (parent[vertex] < 0) continue;
    const root = findRoot(parent, vertex);
    if (!componentMaximumY.has(root)) continue;
    componentMaximumY.set(
      root,
      Math.max(componentMaximumY.get(root), positions[vertex * 3 + 1]),
    );
  }

  const contacts = new Map(
    footRoots.map((root) => [root, { x: 0, y: 0, z: 0, vertexIndices: [] }]),
  );
  for (let vertex = 0; vertex < vertexCount; vertex += 1) {
    if (parent[vertex] < 0) continue;
    const root = findRoot(parent, vertex);
    if (!contacts.has(root)) continue;
    const start = vertex * 3;
    if (
      positions[start + 1] <
      componentMaximumY.get(root) - SOLE_CONTACT_BAND
    ) {
      continue;
    }
    const contact = contacts.get(root);
    contact.x += positions[start];
    contact.y += positions[start + 1];
    contact.z += positions[start + 2];
    contact.vertexIndices.push(vertex);
  }

  const feet = footRoots
    .map((root) => {
      const contact = contacts.get(root);
      const count = contact.vertexIndices.length;
      if (count === 0) throw new Error("第一帧脚底接触区域为空");
      return {
        center: [contact.x / count, contact.y / count, contact.z / count],
        vertexIndices: contact.vertexIndices,
      };
    })
    .sort((first, second) =>
      first.center[0] === second.center[0]
        ? first.center[2] - second.center[2]
        : first.center[0] - second.center[0],
    );
  const footCenters = feet.map((foot) => foot.center);
  return {
    footCenters,
    footVertexIndices: feet.map((foot) => foot.vertexIndices),
    midpoint: footCenters[0].map(
      (value, axis) => (value + footCenters[1][axis]) * 0.5,
    ),
  };
}

export function computeFrameFootCenters(
  positions,
  vertexCount,
  frame,
  footVertexIndices,
) {
  const frameStart = frame * vertexCount * 3;
  if (
    !Number.isInteger(frame) ||
    frame < 0 ||
    frameStart + vertexCount * 3 > positions.length
  ) {
    throw new Error("脚底支撑帧超出网格数据范围");
  }
  return footVertexIndices.map((indices) => {
    if (indices.length === 0) throw new Error("脚底接触顶点为空");
    const center = [0, 0, 0];
    for (const vertex of indices) {
      if (!Number.isInteger(vertex) || vertex < 0 || vertex >= vertexCount) {
        throw new Error("脚底接触顶点超出网格范围");
      }
      const start = frameStart + vertex * 3;
      center[0] += positions[start];
      center[1] += positions[start + 1];
      center[2] += positions[start + 2];
    }
    return center.map((value) => value / indices.length);
  });
}
