import { computeFirstFrameFootSupport } from "./ground-calibration.js";

const USER_COLOR = 0x5de2d6;
const REFERENCE_COLOR = 0xff816d;
const UNIFIED_X = 0.9;
const UNIFIED_GROUND_Y = -1.25;
const GROUND_DISC_RADIUS = 0.78;
const GROUND_DISC_HEIGHT = 0.035;
const UNIFIED_CAMERA_Z = 8.2;
const OVERLAY_TRANSITION_X = 1.15;
const TRANSITION_MS = 900;
const BODY_YAW_RADIANS_PER_PIXEL = 0.009;

const elements = {
  workspace: document.querySelector("#workspace"),
  canvas: document.querySelector("#scene"),
  userVideo: document.querySelector("#userVideo"),
  referenceVideo: document.querySelector("#referenceVideo"),
  threeToggle: document.querySelector("#threeToggle"),
  unifyButton: document.querySelector("#unifyButton"),
  statusText: document.querySelector("#statusText"),
  frameStatus: document.querySelector("#frameStatus"),
  modeNote: document.querySelector("#modeNote"),
  userLabel: document.querySelector("#userLabel"),
  referenceLabel: document.querySelector("#referenceLabel"),
  frameError: document.querySelector("#frameError"),
  meanError: document.querySelector("#meanError"),
  algorithm: document.querySelector("#algorithm"),
  timeline: document.querySelector("#timeline"),
  currentTime: document.querySelector("#currentTime"),
  duration: document.querySelector("#duration"),
  playButton: document.querySelector("#playButton"),
  playIcon: document.querySelector("#playIcon"),
  loadingOverlay: document.querySelector("#loadingOverlay"),
};

const videos = [elements.userVideo, elements.referenceVideo];
const state = {
  metadata: null,
  videoReady: false,
  isPlaying: false,
  scrubbing: false,
  resumeAfterSeek: false,
  seekGeneration: 0,
  frame: -1,
  meshEnabled: false,
  unified: false,
  unifiedProgress: 0,
  unifiedStartedAt: 0,
  unifiedStartProgress: 0,
  unifiedMix: 0,
  unifiedTarget: 0,
  transitionStartedAt: null,
  transitionStartMix: 0,
  sharedBodyYaw: 0,
  bodyYawDrag: null,
  threeLoadVersion: 0,
  threeAbortController: null,
  three: null,
  animationFrameId: null,
  disposed: false,
};

function clampProgress(value) {
  return Math.max(0, Math.min(1, value));
}

function playbackDuration() {
  if (state.videoReady && Number.isFinite(elements.userVideo.duration)) {
    return elements.userVideo.duration;
  }
  return state.metadata?.playback.durationSec ?? 0;
}

function videoProgress() {
  const duration = elements.userVideo.duration;
  if (!state.videoReady || !Number.isFinite(duration) || duration <= 0)
    return 0;
  return clampProgress(elements.userVideo.currentTime / duration);
}

function currentProgress(now = performance.now()) {
  if (!state.unified) return videoProgress();
  if (!state.isPlaying || state.scrubbing) return state.unifiedProgress;
  const duration = Math.max(playbackDuration(), 0.001);
  return Math.min(
    1,
    state.unifiedStartProgress +
      (now - state.unifiedStartedAt) / 1000 / duration,
  );
}

function frameForProgress(progress) {
  const finalFrame = Math.max(1, state.metadata.mesh.stepCount - 1);
  return Math.round(clampProgress(progress) * finalFrame);
}

function updateTimeline(progress) {
  const duration = playbackDuration();
  const frame = frameForProgress(progress);
  elements.timeline.value = String(frame);
  elements.timeline.style.setProperty("--progress", `${progress * 100}%`);
  elements.currentTime.textContent = `${(progress * duration).toFixed(2)} s`;
  elements.frameStatus.textContent = `${(progress * duration).toFixed(2)} / ${duration.toFixed(2)} s`;
}

function updateModeCopy() {
  if (state.unified) {
    elements.statusText.textContent = "统一视角 · 参考为锚";
    elements.modeNote.textContent = "固定基座 · 两个身体绕各自中心同步旋转";
  } else if (state.meshEnabled) {
    elements.statusText.textContent = "视频 + 3D 叠加";
    elements.modeNote.textContent = "3D 锁定视频视角";
  } else {
    elements.statusText.textContent = "双视频同步";
    elements.modeNote.textContent = "双视频同步播放";
  }
}

function updatePlayButton() {
  elements.playIcon.textContent = state.isPlaying ? "Ⅱ" : "▶";
  elements.playButton.setAttribute(
    "aria-label",
    state.isPlaying ? "暂停" : "播放",
  );
}

function pauseVideos() {
  for (const video of videos) video.pause();
}

function setVideoTimes(progress) {
  for (const video of videos) {
    if (!Number.isFinite(video.duration) || video.duration <= 0) continue;
    const safeEnd = Math.max(0, video.duration - 0.001);
    video.currentTime = Math.min(
      safeEnd,
      clampProgress(progress) * video.duration,
    );
  }
}

async function playVideos() {
  if (!state.videoReady || state.unified || state.disposed) return;
  if (videoProgress() >= 0.995) {
    setVideoTimes(0);
    updateTimeline(0);
  }
  try {
    await Promise.all(videos.map((video) => video.play()));
  } catch (error) {
    if (error?.name !== "AbortError") {
      state.isPlaying = false;
      updatePlayButton();
    }
  }
}

function setPlaying(shouldPlay) {
  state.isPlaying = shouldPlay;
  updatePlayButton();
  if (!shouldPlay) {
    pauseVideos();
  } else if (state.unified) {
    if (state.unifiedProgress >= 0.995) state.unifiedProgress = 0;
    state.unifiedStartProgress = state.unifiedProgress;
    state.unifiedStartedAt = performance.now();
  } else {
    void playVideos();
  }
  ensureAnimationLoop();
}

function updateUnifiedMeshFrame(frame) {
  const three = state.three;
  if (
    !three ||
    (three.renderedMode === "unified" && frame === three.renderedFrame)
  ) {
    return;
  }
  const components = state.metadata.mesh.vertexCount * 3;
  const start = frame * components;
  const end = start + components;
  three.userMesh.geometry.attributes.position.array.set(
    three.userPositions.subarray(start, end),
  );
  three.referenceMesh.geometry.attributes.position.array.set(
    three.referencePositions.subarray(start, end),
  );
  three.userMesh.geometry.attributes.position.needsUpdate = true;
  three.referenceMesh.geometry.attributes.position.needsUpdate = true;
  three.renderedMode = "unified";
  three.renderedFrame = frame;
}

function nearestTimestampIndex(timestamps, currentTime) {
  if (timestamps.length <= 1 || currentTime <= timestamps[0]) return 0;
  const last = timestamps.length - 1;
  if (currentTime >= timestamps[last]) return last;
  let low = 0;
  let high = last;
  while (low + 1 < high) {
    const middle = (low + high) >> 1;
    if (timestamps[middle] <= currentTime) low = middle;
    else high = middle;
  }
  return currentTime - timestamps[low] <= timestamps[high] - currentTime
    ? low
    : high;
}

function updateCameraMeshFrames() {
  const three = state.three;
  if (!three) return;
  const components = state.metadata.mesh.vertexCount * 3;
  const overlay = state.metadata.cameraOverlay;
  const userFrame = nearestTimestampIndex(
    overlay.user.timestamps,
    elements.userVideo.currentTime,
  );
  const referenceFrame = nearestTimestampIndex(
    overlay.reference.timestamps,
    elements.referenceVideo.currentTime,
  );
  if (
    three.renderedMode === "camera" &&
    three.renderedUserFrame === userFrame &&
    three.renderedReferenceFrame === referenceFrame
  ) {
    return;
  }

  const userStart = userFrame * components;
  const referenceStart = referenceFrame * components;
  three.userMesh.geometry.attributes.position.array.set(
    three.userCameraPositions.subarray(userStart, userStart + components),
  );
  three.referenceMesh.geometry.attributes.position.array.set(
    three.referenceCameraPositions.subarray(
      referenceStart,
      referenceStart + components,
    ),
  );
  three.userMesh.geometry.attributes.position.needsUpdate = true;
  three.referenceMesh.geometry.attributes.position.needsUpdate = true;
  const userViewport = videoContentViewport(
    elements.userVideo,
    overlay.user.imageSizeHW,
  );
  const referenceViewport = videoContentViewport(
    elements.referenceVideo,
    overlay.reference.imageSizeHW,
  );
  const userHeight = projectedMeshHeight(
    three.userCameraPositions,
    userStart,
    userStart + components,
    three.userCameraIntrinsics,
    three.userCameraTranslation,
    userFrame,
  );
  const referenceHeight = projectedMeshHeight(
    three.referenceCameraPositions,
    referenceStart,
    referenceStart + components,
    three.referenceCameraIntrinsics,
    three.referenceCameraTranslation,
    referenceFrame,
  );
  const userScreenHeight =
    userHeight * (userViewport.height / overlay.user.imageSizeHW[0]);
  const referenceScreenHeight =
    referenceHeight *
    (referenceViewport.height / overlay.reference.imageSizeHW[0]);
  three.overlayUserToReferenceScale = Math.max(
    0.2,
    Math.min(5, userScreenHeight / Math.max(referenceScreenHeight, 0.001)),
  );
  three.renderedMode = "camera";
  three.renderedUserFrame = userFrame;
  three.renderedReferenceFrame = referenceFrame;
}

function projectedMeshHeight(
  positions,
  start,
  end,
  intrinsics,
  translation,
  frame,
) {
  const intrinsicStart = frame * 9;
  const translationStart = frame * 3;
  const focalY = intrinsics[intrinsicStart + 4];
  const centerY = intrinsics[intrinsicStart + 5];
  const translationY = translation[translationStart + 1];
  const translationZ = translation[translationStart + 2];
  let minimum = Infinity;
  let maximum = -Infinity;
  for (let index = start; index < end; index += 3) {
    const depth = positions[index + 2] + translationZ;
    const projectedY =
      (focalY * (positions[index + 1] + translationY)) / depth + centerY;
    minimum = Math.min(minimum, projectedY);
    maximum = Math.max(maximum, projectedY);
  }
  return maximum - minimum;
}

function buildGroundCalibration(
  THREE,
  positions,
  indices,
  vertexCount,
  overlayQuaternion,
) {
  const support = computeFirstFrameFootSupport(positions, indices, vertexCount);
  const firstFoot = new THREE.Vector3()
    .fromArray(support.footCenters[0])
    .applyQuaternion(overlayQuaternion);
  const secondFoot = new THREE.Vector3()
    .fromArray(support.footCenters[1])
    .applyQuaternion(overlayQuaternion);
  const footLine = secondFoot.clone().sub(firstFoot).normalize();
  const horizontalFootLine = footLine.clone();
  horizontalFootLine.y = 0;
  if (horizontalFootLine.lengthSq() < 1e-8) {
    horizontalFootLine.set(1, 0, 0);
  } else {
    horizontalFootLine.normalize();
  }
  const levelingQuaternion = new THREE.Quaternion().setFromUnitVectors(
    footLine,
    horizontalFootLine,
  );
  return {
    footCenter: new THREE.Vector3().fromArray(support.midpoint),
    levelingQuaternion,
    levelingAngle: footLine.angleTo(horizontalFootLine),
  };
}

function setBodyFacing(basisValues, frame, baseQuaternion, target) {
  const start = frame * 9;
  target
    .set(basisValues[start + 2], basisValues[start + 5], basisValues[start + 8])
    .applyQuaternion(baseQuaternion);
  target.y = 0;
  return target.normalize();
}

function updateModelTransform() {
  const three = state.three;
  if (!three) return;
  const overlayQuaternion = three.overlayQuaternion;
  three.userLevelingMix.slerpQuaternions(
    three.identityQuaternion,
    three.userGroundCalibration.levelingQuaternion,
    state.unifiedMix,
  );
  three.referenceLevelingMix.slerpQuaternions(
    three.identityQuaternion,
    three.referenceGroundCalibration.levelingQuaternion,
    state.unifiedMix,
  );
  three.userBaseQuaternion
    .copy(three.userLevelingMix)
    .multiply(overlayQuaternion);
  three.referenceBaseQuaternion
    .copy(three.referenceLevelingMix)
    .multiply(overlayQuaternion);
  setBodyFacing(
    three.userBasis,
    state.frame,
    three.userBaseQuaternion,
    three.userFacing,
  );
  setBodyFacing(
    three.referenceBasis,
    state.frame,
    three.referenceBaseQuaternion,
    three.referenceFacing,
  );
  three.facingCross.crossVectors(three.userFacing, three.referenceFacing);
  const targetAlignmentYaw = Math.atan2(
    three.facingCross.y,
    three.userFacing.dot(three.referenceFacing),
  );
  three.alignmentQuaternion.setFromAxisAngle(
    three.verticalAxis,
    targetAlignmentYaw * state.unifiedMix,
  );
  three.sharedYawQuaternion.setFromAxisAngle(
    three.verticalAxis,
    state.sharedBodyYaw,
  );
  three.userGroup.quaternion
    .copy(three.sharedYawQuaternion)
    .multiply(three.alignmentQuaternion)
    .multiply(three.userBaseQuaternion);
  three.referenceGroup.quaternion
    .copy(three.sharedYawQuaternion)
    .multiply(three.referenceBaseQuaternion);

  const userScale =
    three.overlayUserToReferenceScale +
    (1 - three.overlayUserToReferenceScale) * state.unifiedMix;
  three.userGroup.scale.setScalar(userScale);
  three.referenceGroup.scale.setScalar(1);

  three.userFootCenterWorld
    .copy(three.userGroundCalibration.footCenter)
    .multiplyScalar(userScale)
    .applyQuaternion(three.userGroup.quaternion);
  three.referenceFootCenterWorld
    .copy(three.referenceGroundCalibration.footCenter)
    .applyQuaternion(three.referenceGroup.quaternion);
  three.userGroundTarget
    .set(-UNIFIED_X, UNIFIED_GROUND_Y, 0)
    .sub(three.userFootCenterWorld);
  three.referenceGroundTarget
    .set(UNIFIED_X, UNIFIED_GROUND_Y, 0)
    .sub(three.referenceFootCenterWorld);
  three.userTransitionOrigin.set(-OVERLAY_TRANSITION_X, 0, 0);
  three.referenceTransitionOrigin.set(OVERLAY_TRANSITION_X, 0, 0);
  three.userGroup.position.lerpVectors(
    three.userTransitionOrigin,
    three.userGroundTarget,
    state.unifiedMix,
  );
  three.referenceGroup.position.lerpVectors(
    three.referenceTransitionOrigin,
    three.referenceGroundTarget,
    state.unifiedMix,
  );
}

function setFrame(frame) {
  if (!state.metadata) return;
  const finalFrame = state.metadata.mesh.stepCount - 1;
  const nextFrame = Math.max(0, Math.min(finalFrame, Math.round(frame)));
  if (nextFrame === state.frame) return;
  state.frame = nextFrame;
  const step = state.metadata.alignment.steps[state.frame];
  elements.frameError.textContent = step.distance.toFixed(3);
}

function resizeRenderer() {
  const three = state.three;
  if (!three) return;
  const width = Math.max(1, elements.canvas.clientWidth);
  const height = Math.max(1, elements.canvas.clientHeight);
  three.renderer.setSize(width, height, false);
}

function resetUnifiedView() {
  const three = state.three;
  if (!three) return;
  state.sharedBodyYaw = 0;
  state.bodyYawDrag = null;
  three.camera.position.set(0, 0.15, UNIFIED_CAMERA_Z);
  three.camera.up.set(0, 1, 0);
  three.camera.lookAt(0, 0.05, 0);
  three.camera.updateProjectionMatrix();
  ensureAnimationLoop();
}

function beginBodyYawDrag(event) {
  if (!state.unified || event.button !== 0) return;
  state.bodyYawDrag = {
    pointerId: event.pointerId,
    startX: event.clientX,
    startYaw: state.sharedBodyYaw,
  };
  elements.canvas.setPointerCapture(event.pointerId);
  event.preventDefault();
}

function updateBodyYawDrag(event) {
  const drag = state.bodyYawDrag;
  if (!state.unified || !drag || drag.pointerId !== event.pointerId) return;
  state.sharedBodyYaw =
    drag.startYaw + (event.clientX - drag.startX) * BODY_YAW_RADIANS_PER_PIXEL;
  ensureAnimationLoop();
  event.preventDefault();
}

function endBodyYawDrag(event) {
  const drag = state.bodyYawDrag;
  if (!drag || drag.pointerId !== event.pointerId) return;
  state.bodyYawDrag = null;
  if (elements.canvas.hasPointerCapture(event.pointerId)) {
    elements.canvas.releasePointerCapture(event.pointerId);
  }
  event.preventDefault();
}

function videoContentViewport(video, sourceImageSizeHW) {
  const canvasBounds = elements.canvas.getBoundingClientRect();
  const videoBounds = video.getBoundingClientRect();
  const [sourceHeight, sourceWidth] = sourceImageSizeHW;
  const scale = Math.min(
    videoBounds.width / sourceWidth,
    videoBounds.height / sourceHeight,
  );
  const width = sourceWidth * scale;
  const height = sourceHeight * scale;
  const left = videoBounds.left + (videoBounds.width - width) * 0.5;
  const top = videoBounds.top + (videoBounds.height - height) * 0.5;
  return {
    x: left - canvasBounds.left,
    y: canvasBounds.bottom - (top + height),
    width,
    height,
  };
}

function configureSourceCamera(subject, frame) {
  const three = state.three;
  const source = state.metadata.cameraOverlay[subject];
  const intrinsics =
    subject === "user"
      ? three.userCameraIntrinsics
      : three.referenceCameraIntrinsics;
  const translation =
    subject === "user"
      ? three.userCameraTranslation
      : three.referenceCameraTranslation;
  const intrinsicStart = frame * 9;
  const translationStart = frame * 3;
  const focalX = intrinsics[intrinsicStart];
  const focalY = intrinsics[intrinsicStart + 4];
  const centerX = intrinsics[intrinsicStart + 2];
  const centerY = intrinsics[intrinsicStart + 5];
  const [sourceHeight, sourceWidth] = source.imageSizeHW;
  // SAM-3D-Body's renderer rotates the mesh 180 degrees around X and uses
  // [-cam_t.x, cam_t.y, cam_t.z] as the camera pose. This is equivalent to
  // projecting vertices + cam_t with the stored pinhole intrinsics.
  three.camera.position.set(
    -translation[translationStart],
    translation[translationStart + 1],
    translation[translationStart + 2],
  );
  three.camera.quaternion.identity();
  const near = three.camera.near;
  const left = (-centerX * near) / focalX;
  const right = ((sourceWidth - centerX) * near) / focalX;
  const top = (centerY * near) / focalY;
  const bottom = (-(sourceHeight - centerY) * near) / focalY;
  three.camera.projectionMatrix.makePerspective(
    left,
    right,
    top,
    bottom,
    near,
    three.camera.far,
    three.renderer.coordinateSystem,
  );
  three.camera.projectionMatrixInverse
    .copy(three.camera.projectionMatrix)
    .invert();
}

function renderThree() {
  if (!state.three || !state.meshEnabled) return;
  const three = state.three;
  const width = Math.max(1, elements.canvas.clientWidth);
  const height = Math.max(1, elements.canvas.clientHeight);
  const isVideoOverlay = !state.unified && state.unifiedMix <= 0.001;

  if (!isVideoOverlay) {
    updateUnifiedMeshFrame(state.frame);
    updateModelTransform();
    three.renderer.setScissorTest(false);
    three.renderer.setViewport(0, 0, width, height);
    three.renderer.autoClear = true;
    three.camera.aspect = width / height;
    three.camera.fov = 30;
    three.camera.updateProjectionMatrix();
    three.userGroup.visible = true;
    three.referenceGroup.visible = true;
    three.groundGroup.visible = true;
    three.userGroundDisc.material.opacity = 0.2 * state.unifiedMix;
    three.referenceGroundDisc.material.opacity = 0.2 * state.unifiedMix;
    three.renderer.render(three.scene, three.camera);
    return;
  }

  updateCameraMeshFrames();
  three.userGroup.position.set(0, 0, 0);
  three.referenceGroup.position.set(0, 0, 0);
  three.userGroup.scale.setScalar(1);
  three.referenceGroup.scale.setScalar(1);
  three.userGroup.quaternion.copy(three.overlayQuaternion);
  three.referenceGroup.quaternion.copy(three.overlayQuaternion);
  three.groundGroup.visible = false;

  three.renderer.autoClear = false;
  three.renderer.setScissorTest(false);
  three.renderer.setViewport(0, 0, width, height);
  three.renderer.clear(true, true, true);
  three.renderer.setScissorTest(true);

  const userViewport = videoContentViewport(
    elements.userVideo,
    state.metadata.cameraOverlay.user.imageSizeHW,
  );
  configureSourceCamera("user", three.renderedUserFrame);
  three.userGroup.visible = true;
  three.referenceGroup.visible = false;
  three.renderer.setViewport(
    userViewport.x,
    userViewport.y,
    userViewport.width,
    userViewport.height,
  );
  three.renderer.setScissor(
    userViewport.x,
    userViewport.y,
    userViewport.width,
    userViewport.height,
  );
  three.renderer.render(three.scene, three.camera);

  const referenceViewport = videoContentViewport(
    elements.referenceVideo,
    state.metadata.cameraOverlay.reference.imageSizeHW,
  );
  configureSourceCamera("reference", three.renderedReferenceFrame);
  three.userGroup.visible = false;
  three.referenceGroup.visible = true;
  three.renderer.setViewport(
    referenceViewport.x,
    referenceViewport.y,
    referenceViewport.width,
    referenceViewport.height,
  );
  three.renderer.setScissor(
    referenceViewport.x,
    referenceViewport.y,
    referenceViewport.width,
    referenceViewport.height,
  );
  three.renderer.render(three.scene, three.camera);

  three.userGroup.visible = true;
  three.renderer.setScissorTest(false);
  three.renderer.autoClear = true;
}

function animate(now) {
  state.animationFrameId = null;
  if (state.disposed) return;

  if (!state.scrubbing) {
    const progress = currentProgress(now);
    if (state.unified) state.unifiedProgress = progress;
    updateTimeline(progress);
    setFrame(frameForProgress(progress));
    if (state.unified && state.isPlaying && progress >= 1) {
      state.isPlaying = false;
      updatePlayButton();
    }
  }

  let transitionActive = false;
  if (state.transitionStartedAt !== null) {
    const elapsed = Math.min(
      1,
      (now - state.transitionStartedAt) / TRANSITION_MS,
    );
    const eased = 1 - Math.pow(1 - elapsed, 3);
    state.unifiedMix =
      state.transitionStartMix +
      (state.unifiedTarget - state.transitionStartMix) * eased;
    transitionActive = elapsed < 1;
    if (!transitionActive) {
      state.unifiedMix = state.unifiedTarget;
      state.transitionStartedAt = null;
    }
  }

  renderThree();
  if (state.isPlaying || transitionActive) ensureAnimationLoop();
}

function ensureAnimationLoop() {
  if (state.animationFrameId === null && !state.disposed) {
    state.animationFrameId = requestAnimationFrame(animate);
  }
}

async function fetchBinary(path, Type, signal) {
  const response = await fetch(path, { signal });
  if (!response.ok) throw new Error(`加载失败: ${path} (${response.status})`);
  return new Type(await response.arrayBuffer());
}

function buildMesh(
  THREE,
  indices,
  color,
  { wireframe = false, opacity = 0.6 } = {},
) {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute(
    "position",
    new THREE.BufferAttribute(
      new Float32Array(state.metadata.mesh.vertexCount * 3),
      3,
    ),
  );
  geometry.setIndex(new THREE.BufferAttribute(indices, 1));
  const material = new THREE.MeshBasicMaterial({
    color,
    wireframe,
    transparent: true,
    opacity,
    depthTest: true,
    depthWrite: false,
    side: THREE.DoubleSide,
  });
  const mesh = new THREE.Mesh(geometry, material);
  mesh.frustumCulled = false;
  return mesh;
}

function buildDepthOccluder(THREE, geometry) {
  const material = new THREE.MeshBasicMaterial({
    colorWrite: false,
    depthTest: true,
    depthWrite: true,
    side: THREE.DoubleSide,
  });
  const mesh = new THREE.Mesh(geometry, material);
  mesh.frustumCulled = false;
  return mesh;
}

function buildGroundDisc(THREE, color, x) {
  const geometry = new THREE.CylinderGeometry(
    GROUND_DISC_RADIUS,
    GROUND_DISC_RADIUS,
    GROUND_DISC_HEIGHT,
    64,
  );
  const material = new THREE.MeshBasicMaterial({
    color,
    transparent: true,
    opacity: 0,
    depthWrite: false,
    side: THREE.DoubleSide,
  });
  const disc = new THREE.Mesh(geometry, material);
  disc.position.set(x, UNIFIED_GROUND_Y - GROUND_DISC_HEIGHT * 0.5, 0);
  disc.frustumCulled = false;
  return disc;
}

function createCompatibleWebGLContext(canvas) {
  const attributes = {
    alpha: true,
    antialias: false,
    depth: true,
    powerPreference: "low-power",
    premultipliedAlpha: true,
  };
  const context = canvas.getContext("webgl2", attributes);
  if (!context) throw new Error("当前浏览器无法创建 WebGL2 上下文");

  const nativeGetShaderPrecisionFormat =
    context.getShaderPrecisionFormat.bind(context);
  let usedPrecisionFallback = false;
  const compatibleGetShaderPrecisionFormat = (shaderType, precisionType) => {
    const format = nativeGetShaderPrecisionFormat(shaderType, precisionType);
    if (format) return format;
    usedPrecisionFallback = true;
    return { rangeMin: 127, rangeMax: 127, precision: 23 };
  };

  // Some embedded Chromium builds expose WebGL2 but return null for this
  // mandatory query. Three.js dereferences it during capability detection.
  Object.defineProperty(context, "getShaderPrecisionFormat", {
    configurable: true,
    value: compatibleGetShaderPrecisionFormat,
  });
  compatibleGetShaderPrecisionFormat(context.VERTEX_SHADER, context.HIGH_FLOAT);
  compatibleGetShaderPrecisionFormat(
    context.FRAGMENT_SHADER,
    context.HIGH_FLOAT,
  );
  return {
    context,
    attributes,
    usedPrecisionFallback: () => usedPrecisionFallback,
  };
}

async function initializeThree() {
  if (state.three) return;
  const loadVersion = ++state.threeLoadVersion;
  state.threeAbortController?.abort();
  const controller = new AbortController();
  state.threeAbortController = controller;
  elements.threeToggle.disabled = true;
  elements.statusText.textContent = "正在按需加载 3D";

  try {
    const THREE = await import("three");
    const base = "/data/";
    const files = state.metadata.files;
    const [
      userPositions,
      referencePositions,
      userBasis,
      referenceBasis,
      userCameraPositions,
      referenceCameraPositions,
      userCameraTranslation,
      referenceCameraTranslation,
      userCameraIntrinsics,
      referenceCameraIntrinsics,
      indices,
    ] = await Promise.all([
      fetchBinary(
        base + files.userRawPositions,
        Float32Array,
        controller.signal,
      ),
      fetchBinary(
        base + files.referenceRawPositions,
        Float32Array,
        controller.signal,
      ),
      fetchBinary(base + files.userBasis, Float32Array, controller.signal),
      fetchBinary(base + files.referenceBasis, Float32Array, controller.signal),
      fetchBinary(
        base + files.userCameraPositions,
        Float32Array,
        controller.signal,
      ),
      fetchBinary(
        base + files.referenceCameraPositions,
        Float32Array,
        controller.signal,
      ),
      fetchBinary(
        base + files.userCameraTranslation,
        Float32Array,
        controller.signal,
      ),
      fetchBinary(
        base + files.referenceCameraTranslation,
        Float32Array,
        controller.signal,
      ),
      fetchBinary(
        base + files.userCameraIntrinsics,
        Float32Array,
        controller.signal,
      ),
      fetchBinary(
        base + files.referenceCameraIntrinsics,
        Float32Array,
        controller.signal,
      ),
      fetchBinary(base + files.indices, Uint32Array, controller.signal),
    ]);
    if (loadVersion !== state.threeLoadVersion || !state.meshEnabled) return;

    const compatibleContext = createCompatibleWebGLContext(elements.canvas);
    const renderer = new THREE.WebGLRenderer({
      canvas: elements.canvas,
      context: compatibleContext.context,
      alpha: true,
      antialias: false,
      powerPreference: "low-power",
    });
    renderer.setClearColor(0x000000, 0);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
    renderer.outputColorSpace = THREE.SRGBColorSpace;

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(30, 1, 0.01, 40);
    camera.position.set(0, 0.15, UNIFIED_CAMERA_Z);
    camera.lookAt(0, 0.05, 0);
    const handleBodyYawReset = () => {
      if (state.unified) resetUnifiedView();
    };
    elements.canvas.addEventListener("pointerdown", beginBodyYawDrag);
    elements.canvas.addEventListener("pointermove", updateBodyYawDrag);
    elements.canvas.addEventListener("pointerup", endBodyYawDrag);
    elements.canvas.addEventListener("pointercancel", endBodyYawDrag);
    elements.canvas.addEventListener("dblclick", handleBodyYawReset);

    const userGroup = new THREE.Group();
    const referenceGroup = new THREE.Group();
    const groundGroup = new THREE.Group();
    const userMesh = buildMesh(THREE, indices, USER_COLOR, {
      wireframe: true,
      opacity: 0.66,
    });
    const referenceMesh = buildMesh(THREE, indices, REFERENCE_COLOR, {
      wireframe: true,
      opacity: 0.66,
    });
    const userDepthOccluder = buildDepthOccluder(THREE, userMesh.geometry);
    const referenceDepthOccluder = buildDepthOccluder(
      THREE,
      referenceMesh.geometry,
    );
    userGroup.add(userDepthOccluder, userMesh);
    referenceGroup.add(referenceDepthOccluder, referenceMesh);
    const userGroundDisc = buildGroundDisc(THREE, USER_COLOR, -UNIFIED_X);
    const referenceGroundDisc = buildGroundDisc(
      THREE,
      REFERENCE_COLOR,
      UNIFIED_X,
    );
    groundGroup.add(userGroundDisc, referenceGroundDisc);
    scene.add(groundGroup, userGroup, referenceGroup);
    const overlayQuaternion = new THREE.Quaternion().setFromAxisAngle(
      new THREE.Vector3(1, 0, 0),
      Math.PI,
    );
    const userGroundCalibration = buildGroundCalibration(
      THREE,
      userPositions,
      indices,
      state.metadata.mesh.vertexCount,
      overlayQuaternion,
    );
    const referenceGroundCalibration = buildGroundCalibration(
      THREE,
      referencePositions,
      indices,
      state.metadata.mesh.vertexCount,
      overlayQuaternion,
    );

    state.three = {
      THREE,
      renderer,
      scene,
      camera,
      handleBodyYawReset,
      userGroup,
      referenceGroup,
      groundGroup,
      userMesh,
      referenceMesh,
      userDepthOccluder,
      referenceDepthOccluder,
      userGroundDisc,
      referenceGroundDisc,
      userPositions,
      referencePositions,
      userBasis,
      referenceBasis,
      userCameraPositions,
      referenceCameraPositions,
      userCameraTranslation,
      referenceCameraTranslation,
      userCameraIntrinsics,
      referenceCameraIntrinsics,
      usedPrecisionFallback: compatibleContext.usedPrecisionFallback(),
      overlayQuaternion,
      verticalAxis: new THREE.Vector3(0, 1, 0),
      userFacing: new THREE.Vector3(),
      referenceFacing: new THREE.Vector3(),
      facingCross: new THREE.Vector3(),
      alignmentQuaternion: new THREE.Quaternion(),
      sharedYawQuaternion: new THREE.Quaternion(),
      identityQuaternion: new THREE.Quaternion(),
      userLevelingMix: new THREE.Quaternion(),
      referenceLevelingMix: new THREE.Quaternion(),
      userBaseQuaternion: new THREE.Quaternion(),
      referenceBaseQuaternion: new THREE.Quaternion(),
      userGroundCalibration,
      referenceGroundCalibration,
      userFootCenterWorld: new THREE.Vector3(),
      referenceFootCenterWorld: new THREE.Vector3(),
      userGroundTarget: new THREE.Vector3(),
      referenceGroundTarget: new THREE.Vector3(),
      userTransitionOrigin: new THREE.Vector3(),
      referenceTransitionOrigin: new THREE.Vector3(),
      overlayUserToReferenceScale: 1,
      renderedMode: null,
      renderedFrame: -1,
      renderedUserFrame: -1,
      renderedReferenceFrame: -1,
    };
    resizeRenderer();
    renderThree();
  } finally {
    if (loadVersion === state.threeLoadVersion) {
      elements.threeToggle.disabled = false;
      updateModeCopy();
    }
  }
}

function destroyThree() {
  state.threeLoadVersion += 1;
  state.threeAbortController?.abort();
  state.threeAbortController = null;
  const three = state.three;
  if (!three) return;
  elements.canvas.removeEventListener("pointerdown", beginBodyYawDrag);
  elements.canvas.removeEventListener("pointermove", updateBodyYawDrag);
  elements.canvas.removeEventListener("pointerup", endBodyYawDrag);
  elements.canvas.removeEventListener("pointercancel", endBodyYawDrag);
  elements.canvas.removeEventListener("dblclick", three.handleBodyYawReset);
  for (const mesh of [
    three.userMesh,
    three.referenceMesh,
    three.userGroundDisc,
    three.referenceGroundDisc,
  ]) {
    mesh.geometry.dispose();
    mesh.material.dispose();
  }
  three.userDepthOccluder.material.dispose();
  three.referenceDepthOccluder.material.dispose();
  three.scene.clear();
  three.renderer.dispose();
  three.renderer.forceContextLoss();
  const replacementCanvas = elements.canvas.cloneNode(false);
  replacementCanvas.width = 1;
  replacementCanvas.height = 1;
  elements.canvas.replaceWith(replacementCanvas);
  elements.canvas = replacementCanvas;
  state.three = null;
}

function startUnifiedTransition(target) {
  state.unifiedTarget = target;
  state.transitionStartMix = state.unifiedMix;
  state.transitionStartedAt = performance.now();
  ensureAnimationLoop();
}

async function setMeshEnabled(enabled) {
  if (enabled === state.meshEnabled) return;
  state.meshEnabled = enabled;
  elements.workspace.classList.toggle("mesh-enabled", enabled);
  elements.threeToggle.setAttribute("aria-checked", String(enabled));
  elements.unifyButton.disabled = !enabled;

  if (!enabled) {
    if (state.unified) await setUnified(false);
    destroyThree();
    updateModeCopy();
    return;
  }

  try {
    await initializeThree();
    if (state.meshEnabled) renderThree();
  } catch (error) {
    if (error?.name === "AbortError") return;
    console.error(error);
    state.meshEnabled = false;
    elements.workspace.classList.remove("mesh-enabled");
    elements.threeToggle.setAttribute("aria-checked", "false");
    elements.unifyButton.disabled = true;
    elements.statusText.textContent = "3D 加载失败";
  }
}

async function setUnified(enabled) {
  if (enabled === state.unified || (enabled && !state.meshEnabled)) return;
  if (enabled) {
    state.unifiedProgress = videoProgress();
    state.unifiedStartProgress = state.unifiedProgress;
    state.unifiedStartedAt = performance.now();
    pauseVideos();
  } else {
    state.unifiedProgress = currentProgress();
    setVideoTimes(state.unifiedProgress);
  }
  state.unified = enabled;
  if (state.three) resetUnifiedView();
  elements.workspace.classList.toggle("unified-mode", enabled);
  elements.unifyButton.classList.toggle("active", enabled);
  elements.unifyButton.setAttribute("aria-pressed", String(enabled));
  startUnifiedTransition(enabled ? 1 : 0);
  updateModeCopy();
  if (!enabled && state.isPlaying) void playVideos();
}

function seekWait(video, progress) {
  return new Promise((resolve) => {
    if (!Number.isFinite(video.duration) || video.duration <= 0) {
      resolve();
      return;
    }
    const target = Math.min(video.duration - 0.001, progress * video.duration);
    let settled = false;
    let timeoutId;
    const finish = () => {
      if (settled) return;
      settled = true;
      window.clearTimeout(timeoutId);
      video.removeEventListener("seeked", finish);
      resolve();
    };
    video.addEventListener("seeked", finish, { once: true });
    video.currentTime = Math.max(0, target);
    timeoutId = window.setTimeout(finish, 700);
  });
}

function previewSeek(frame) {
  const finalFrame = Math.max(1, state.metadata.mesh.stepCount - 1);
  const progress = clampProgress(frame / finalFrame);
  if (state.unified) {
    state.unifiedProgress = progress;
  } else {
    setVideoTimes(progress);
  }
  updateTimeline(progress);
  setFrame(frame);
  renderThree();
}

async function commitSeek() {
  if (!state.scrubbing) return;
  const generation = ++state.seekGeneration;
  state.scrubbing = false;
  const finalFrame = Math.max(1, state.metadata.mesh.stepCount - 1);
  const progress = clampProgress(Number(elements.timeline.value) / finalFrame);
  if (state.unified) {
    state.unifiedProgress = progress;
  } else {
    await Promise.all(videos.map((video) => seekWait(video, progress)));
  }
  if (generation !== state.seekGeneration) return;
  if (state.unified) {
    state.unifiedStartProgress = progress;
    state.unifiedStartedAt = performance.now();
  }
  if (state.resumeAfterSeek && !state.unified) await playVideos();
  ensureAnimationLoop();
}

function beginScrub() {
  if (state.scrubbing) return;
  state.scrubbing = true;
  state.resumeAfterSeek = state.isPlaying;
  pauseVideos();
}

function waitForVideoMetadata(video) {
  if (video.readyState >= HTMLMediaElement.HAVE_METADATA)
    return Promise.resolve();
  return new Promise((resolve, reject) => {
    video.addEventListener("loadedmetadata", resolve, { once: true });
    video.addEventListener(
      "error",
      () => reject(new Error(`视频加载失败: ${video.currentSrc || video.src}`)),
      { once: true },
    );
  });
}

async function loadViewer() {
  const [metadataResponse] = await Promise.all([
    fetch("/data/viewer-data.json"),
    Promise.all(videos.map(waitForVideoMetadata)),
  ]);
  if (!metadataResponse.ok) throw new Error("找不到 viewer-data.json");
  state.metadata = await metadataResponse.json();
  state.videoReady = true;

  elements.userLabel.textContent = state.metadata.labels.user;
  elements.referenceLabel.textContent = state.metadata.labels.reference;
  elements.algorithm.textContent = String(
    state.metadata.alignment.algorithm || "DTW",
  ).toUpperCase();
  elements.meanError.textContent = Number(
    state.metadata.alignment.summary.meanFrameError,
  ).toFixed(3);
  elements.timeline.max = String(state.metadata.mesh.stepCount - 1);
  elements.duration.textContent = `${playbackDuration().toFixed(2)} s`;
  setFrame(0);
  updateTimeline(0);
  updateModeCopy();
  elements.loadingOverlay.classList.add("hidden");
  setPlaying(true);
}

function releaseResources() {
  if (state.disposed) return;
  state.disposed = true;
  if (state.animationFrameId !== null)
    cancelAnimationFrame(state.animationFrameId);
  state.animationFrameId = null;
  pauseVideos();
  destroyThree();
  for (const video of videos) {
    video.removeAttribute("src");
    video.load();
  }
}

elements.playButton.addEventListener("click", () =>
  setPlaying(!state.isPlaying),
);
elements.threeToggle.addEventListener("click", () => {
  void setMeshEnabled(!state.meshEnabled);
});
elements.unifyButton.addEventListener("click", () => {
  void setUnified(!state.unified);
});
elements.timeline.addEventListener("pointerdown", beginScrub);
elements.timeline.addEventListener("input", (event) => {
  beginScrub();
  previewSeek(Number(event.target.value));
});
elements.timeline.addEventListener("click", (event) => {
  const bounds = elements.timeline.getBoundingClientRect();
  if (bounds.width <= 0) return;
  const progress = clampProgress((event.clientX - bounds.left) / bounds.width);
  const frame = Math.round(progress * (state.metadata.mesh.stepCount - 1));
  beginScrub();
  previewSeek(frame);
  void commitSeek();
});
elements.timeline.addEventListener("change", () => void commitSeek());
elements.timeline.addEventListener("pointerup", () => void commitSeek());
elements.timeline.addEventListener("pointercancel", () => void commitSeek());
elements.userVideo.addEventListener("ended", () => {
  if (!state.unified && state.isPlaying && !state.scrubbing) {
    state.isPlaying = false;
    updatePlayButton();
    pauseVideos();
    updateTimeline(1);
    setFrame(state.metadata.mesh.stepCount - 1);
  }
});
window.addEventListener("resize", () => {
  resizeRenderer();
  renderThree();
});
window.addEventListener("keydown", (event) => {
  if (event.code === "Space") {
    event.preventDefault();
    setPlaying(!state.isPlaying);
  } else if (event.code === "ArrowRight" || event.code === "ArrowLeft") {
    beginScrub();
    const delta = event.code === "ArrowRight" ? 1 : -1;
    const frame = Math.max(
      0,
      Math.min(state.metadata.mesh.stepCount - 1, state.frame + delta),
    );
    previewSeek(frame);
    void commitSeek();
  }
});
document.addEventListener("visibilitychange", () => {
  if (document.hidden && state.isPlaying) setPlaying(false);
});
window.addEventListener("pagehide", releaseResources, { once: true });
window.addEventListener("beforeunload", releaseResources, { once: true });

window.__alignmentViewerDiagnostics = () => ({
  webglActive: Boolean(state.three),
  meshEnabled: state.meshEnabled,
  unified: state.unified,
  sharedBodyYaw: state.sharedBodyYaw,
  firstFrameGrounding: state.three
    ? {
        userLevelingDegrees:
          (state.three.userGroundCalibration.levelingAngle * 180) / Math.PI,
        referenceLevelingDegrees:
          (state.three.referenceGroundCalibration.levelingAngle * 180) /
          Math.PI,
      }
    : null,
  temporalFilter: state.metadata?.temporalFilter?.type ?? "none",
  animationScheduled: state.animationFrameId !== null,
  frame: state.frame,
  userTime: elements.userVideo.currentTime,
  referenceTime: elements.referenceVideo.currentTime,
  renderBackend: state.three
    ? state.three.usedPrecisionFallback
      ? "webgl2-precision-fallback"
      : "webgl2"
    : "none",
  renderCalls: state.three?.renderer.info.render.calls ?? 0,
  renderedTriangles: state.three?.renderer.info.render.triangles ?? 0,
});

pauseVideos();
loadViewer().catch((error) => {
  console.error(error);
  elements.statusText.textContent = "加载失败";
  elements.loadingOverlay.querySelector("p").textContent = error.message;
});
