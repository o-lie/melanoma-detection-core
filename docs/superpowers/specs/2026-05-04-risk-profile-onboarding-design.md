# Risk Profile Onboarding — Design Spec

**Date:** 2026-05-04  
**Status:** Approved

## Overview

Extend the melanoma detection app with a mandatory onboarding wizard that collects clinical risk factors. These factors are sent to the AI API alongside every image analysis, where they dynamically adjust the decision threshold — producing a clinically sounder result than image-only classification.

## Architecture & Flow

```
Disclaimer → [onboarding_completed in AsyncStorage?]
               → NO  → OnboardingWizard (fullscreen) → Tabs
               → YES → Tabs
```

Flag `onboarding_completed` stored in AsyncStorage, checked in `_layout.tsx` after `initDatabase()`. If missing → `router.replace('/onboarding')`.

After onboarding save → `router.replace('/(tabs)')`.  
Profile edit → push from Profile tab → after save → `router.back()`.

### New files / changes

| File | Change |
|------|--------|
| `app/onboarding.tsx` | New route — fullscreen wizard, no tabs |
| `components/ProfileWizard.tsx` | Shared multi-step form, `mode: "onboarding" \| "edit"` prop |
| `db/user-profile.repository.ts` | CRUD for `user_profile` table |
| `app/(tabs)/profile.tsx` | Add risk profile section + Edit button |
| `app/result.tsx` | Show `clinical_risk_level` and adjusted threshold info |
| `melanoma-detection-ai/api/main.py` | Extend `/predict` with risk profile |

## Onboarding Wizard — Steps

4 steps with progress bar. Navigation: Next / Back. DB write only on step 4 completion.

All fields optional. Step 4 submission requires at least age OR skin_phototype to be set.

**Step 1 — Basic info**
- Age (numeric input)
- Skin phototype Fitzpatrick I–VI (picker with color sample + description per type)

**Step 2 — Family history**
- Family history of skin cancer? (toggle)
  - If yes → who? (optional free text)
- Family history of other cancers? (toggle)
  - If yes → who? (optional free text)

**Step 3 — Lifestyle**
- Severe sunburns in the past? (toggle)
- Frequent sun exposure? (toggle)
- Tanning bed use? (toggle)

**Step 4 — Skin appearance**
- Many moles (>50)? (toggle)
- Atypical moles (irregular, large)? (toggle)
- Very fair skin? (toggle)

## API — `/predict` Extension

Endpoint accepts `multipart/form-data`. New optional field `risk_profile` (JSON string).

```python
class RiskProfile(BaseModel):
    age: int | None = None
    skin_phototype: str | None = None        # "I"–"VI" Fitzpatrick
    family_history_skin_cancer: bool = False
    family_history_other_cancer: bool = False
    had_severe_sunburns: bool = False
    frequent_sun_exposure: bool = False
    uses_tanning_beds: bool = False
    many_moles: bool = False
    atypical_moles: bool = False
    very_fair_skin: bool = False
```

### Risk scoring → threshold

Each factor carries a point weight. Points summed → `risk_score` → mapped to level.

| Factor | Points |
|--------|--------|
| family_history_skin_cancer | 3 |
| atypical_moles | 3 |
| family_history_other_cancer | 2 |
| many_moles | 2 |
| skin_phototype I or II | 2 |
| had_severe_sunburns | 1 |
| uses_tanning_beds | 1 |
| frequent_sun_exposure | 1 |
| very_fair_skin | 1 |
| age > 50 | 1 |

| risk_score | clinical_risk_level | threshold_used |
|---|---|---|
| 0–2 | low | 0.50 |
| 3–5 | medium | 0.42 |
| ≥ 6 | high | 0.35 |

If no risk profile provided → fallback threshold 0.50, `clinical_risk_level: null`.

### Response

```json
{
  "probability": 0.41,
  "threshold_used": 0.35,
  "clinical_risk_level": "high",
  "label": "high_risk",
  "disclaimer": "This is not a medical diagnosis. Consult a dermatologist."
}
```

## Profile Tab — After Onboarding

### Risk Profile section
- `clinical_risk_level` badge (color-coded: green / yellow / red)
- List of active risk factors (icon + label)
- "Edit profile" button → `ProfileWizard` in `edit` mode

### Result screen changes
- Show `clinical_risk_level` from latest analysis
- If `threshold_used` != 0.50 → info banner: "Threshold adjusted for your risk profile"
- Raw probability still visible

### Sending profile to API
- `user-profile.repository` fetches profile from SQLite before every upload
- Profile serialized as JSON string, added to `FormData` as `risk_profile` field

## Data Model

`user_profile` table already exists in `db/database.ts` with all required columns. No schema migration needed.

New repository `db/user-profile.repository.ts` needs:
- `upsertUserProfile(profile)` — INSERT OR REPLACE with `id = 1`
- `getUserProfile()` — SELECT WHERE id = 1
- `hasUserProfile()` — returns bool (for onboarding check fallback)