# Pose Correction MVP Scope Definition

**Version:** 1.0
**Date:** 2026-02-24
**Status:** Draft for Review

## 1. Executive Summary

This document defines the Minimum Viable Product (MVP) scope for the Pose Correction feature in Duolian. The MVP focuses on delivering a high-quality single-action analysis before expanding to multiple techniques.

## 2. MVP Scope

### 2.1 Supported Action

| Phase | Action | Chinese Name | Priority |
|-------|--------|--------------|----------|
| MVP | **Smash** | 杀球 | P0 |
| Post-MVP | Clear | 高远球 | P1 |
| Post-MVP | Drop Shot | 吊球 | P1 |
| Post-MVP | Net Shot | 网前球 | P2 |
| Future | Drive | 平抽球 | P2 |
| Future | All techniques | 全部技术 | P3 |

**Rationale for starting with Smash:**
- Most visually distinctive motion with clear phases
- High user demand (offensive technique)
- Clear reference material available from professional players
- Easier to validate correctness due to explosive nature

### 2.2 Reference Template

| Aspect | MVP Specification |
|--------|-------------------|
| Templates per action | 1 (professional player reference) |
| Player handedness | Right-handed only |
| Player gender | Male professional player |
| Camera angle | Rear/side view (玩家背后/侧面视角) |
| Video specs | 1080p@30fps, MP4 format |

**Limitations:**
- Left-handed players will receive analysis but alignment may be suboptimal
- Non-standard camera angles may produce less accurate results

### 2.3 Camera Position

```
MVP Support:
├── Rear view (背后视角) ✓ PRIMARY
└── Side view (侧面视角) ✓ SECONDARY

NOT in MVP:
├── Front view (正面视角)
├── Overhead view (顶视角)
└── Multi-angle fusion
```

**Recommended Setup:**
- Camera positioned behind the player
- Height: 1.2-1.5m (tripod level)
- Distance: 5-8 meters from court
- Frame: Full body visible throughout motion

## 3. Technical Specifications

### 3.1 Input Requirements

| Parameter | Specification |
|-----------|---------------|
| Video format | MP4, MOV |
| Resolution | Minimum 720p, recommended 1080p |
| Frame rate | 30fps (will be resampled if different) |
| Duration | 2-5 seconds per shot |
| Lighting | Well-lit, avoid strong backlight |

### 3.2 Processing Pipeline

```
Input Video
    ↓
Frame Extraction (30fps)
    ↓
Subject Detection (SAM 3D Body)
    ↓
3D Skeleton Estimation (MHR70 format)
    ↓
Temporal Smoothing
    ↓
Skeleton Normalization
    ↓
FastDTW Alignment with Reference
    ↓
Scoring & Tip Generation
    ↓
Results Output
```

### 3.3 Output Format

```typescript
interface PoseCorrectionResult {
  // Metadata
  actionType: "smash";
  processingVersion: "1.0.0";
  timestamp: string;

  // Overall score
  overallScore: number; // 0-100

  // Temporal alignment
  alignmentPath: Array<[userFrame: number, refFrame: number]>;

  // Per-frame analysis
  frameErrors: Array<{
    frameIndex: number;
    timestamp: number;
    jointErrors: Record<string, number>; // joint name -> error in cm
    maxErrorJoint: string;
  }>;

  // Phase analysis (if detectable)
  phases?: {
    backswing: { startFrame: number; endFrame: number; score: number };
    acceleration: { startFrame: number; endFrame: number; score: number };
    contact: { startFrame: number; endFrame: number; score: number };
    followThrough: { startFrame: number; endFrame: number; score: number };
  };

  // Top issues
  topErrors: Array<{
    jointName: string;
    severity: "critical" | "moderate" | "minor";
    description: string;
    correction: string;
    timestamp: number;
  }>;

  // Improvement tips
  tips: string[];
}
```

## 4. Feature Flags

```typescript
// Feature flag configuration for gradual rollout
const POSE_CORRECTION_FLAGS = {
  // Enable pose correction feature in UI
  ENABLE_POSE_CORRECTION: true,

  // MVP: Only smash action
  SUPPORTED_ACTIONS: ["smash"],

  // Enable left-handed support
  SUPPORT_LEFT_HANDED: false,

  // Enable multi-angle support
  SUPPORT_MULTI_ANGLE: false,

  // Enable phase detection
  ENABLE_PHASE_DETECTION: false,

  // Maximum video duration (seconds)
  MAX_VIDEO_DURATION: 10,

  // Maximum file size (MB)
  MAX_FILE_SIZE: 100,
};
```

## 5. Roadmap

### Phase 1: MVP (Current)
- Single action: Smash
- Single reference template
- Rear/side camera view
- Right-handed only

### Phase 2: Actions Expansion (Q2 2026)
- Add Clear and Drop Shot
- Multiple reference templates per action
- Basic left-handed support (mirror alignment)

### Phase 3: Camera Flexibility (Q3 2026)
- Front view support
- Camera angle detection
- Angle-specific reference templates

### Phase 4: Advanced Features (Q4 2026)
- All technique types
- Multi-angle fusion
- Automatic phase detection
- Personalized reference (user's own best shots)

## 6. Success Metrics

| Metric | MVP Target |
|--------|------------|
| Analysis success rate | > 90% |
| Average processing time | < 30 seconds |
| User satisfaction score | > 4.0/5 |
| Error detection accuracy | > 80% (validated vs expert) |

## 7. Limitations & Disclaimers

### Documented Limitations

1. **Single Action Only**: Only smash technique is supported in MVP
2. **Camera Angle**: Best results with rear/side view; other angles may be less accurate
3. **Handedness**: Optimized for right-handed players
4. **Reference Quality**: Results depend on quality of professional reference template
5. **Environmental Factors**: Lighting, background, and occlusion can affect accuracy

### User Communication

Users should be informed:
- This is an automated analysis tool, not a replacement for professional coaching
- Results are most accurate when following recommended setup guidelines
- The system will improve over time with more data

## 8. Open Questions

1. Should we allow users to upload their own reference videos (personal best) in Phase 2?
2. Do we need different reference templates for different skill levels (beginner vs advanced)?
3. Should we support batch analysis of multiple shots in a single video?

## 9. Related Documents

- [TODO.md](../TODO.md) - Task tracking
- [Protocol Definition](./PROTOCOL.md) - Technical protocol (Task #1)
- [Reference Asset Spec](./REFERENCE_ASSET_SPEC.md) - Reference video specifications (Task #2)

---

**Next Steps:**
1. Review and approve this MVP scope
2. Create reference asset metadata specification (Task #2)
3. Procure/record professional smash reference video
4. Begin video inference API implementation (Task #13)
