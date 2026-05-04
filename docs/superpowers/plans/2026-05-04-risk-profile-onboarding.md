# Risk Profile Onboarding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a mandatory 4-step onboarding wizard that collects clinical risk factors, stores them in SQLite, sends them to the AI API on every analysis, and uses them to dynamically adjust the melanoma detection threshold.

**Architecture:** Shared `ProfileWizard` component (mode: onboarding | edit) drives both first-run onboarding and profile editing. AsyncStorage flag `onboarding_completed` controls routing in `_layout.tsx`. Python API extracts risk scoring to a pure function (testable), dynamically adjusts threshold, returns `clinical_risk_level` alongside image probability.

**Tech Stack:** Expo Router, React Native, SQLite (expo-sqlite), AsyncStorage (@react-native-async-storage/async-storage), TypeScript, FastAPI, pytest

---

## File Map

| File | Action | Purpose |
|------|--------|---------|
| `melanoma-detection-app/types/user-profile.ts` | Create | UserProfile type |
| `melanoma-detection-app/db/user-profile.repository.ts` | Create | SQLite CRUD for user_profile |
| `melanoma-detection-app/app/_layout.tsx` | Modify | AsyncStorage check, onboarding route |
| `melanoma-detection-app/components/ProfileWizard.tsx` | Create | 4-step form, mode prop |
| `melanoma-detection-app/app/onboarding.tsx` | Create | Fullscreen onboarding wrapper |
| `melanoma-detection-app/lib/api/predict.ts` | Modify | Send risk_profile in FormData |
| `melanoma-detection-app/app/(tabs)/profile.tsx` | Modify | Show risk profile section + Edit button |
| `melanoma-detection-app/app/result.tsx` | Modify | Show clinical_risk_level + threshold info |
| `melanoma-detection-ai/api/main.py` | Modify | RiskProfile model, scoring logic, /predict |
| `melanoma-detection-ai/api/test_risk_scoring.py` | Create | pytest for scoring function |

---

## Task 1: UserProfile type

**Files:**
- Create: `melanoma-detection-app/types/user-profile.ts`

- [ ] **Step 1: Create the type file**

```typescript
// melanoma-detection-app/types/user-profile.ts
export type SkinPhototype = "I" | "II" | "III" | "IV" | "V" | "VI";

export type ClinicalRiskLevel = "low" | "medium" | "high";

export type UserProfile = {
  id?: number;
  age: number | null;
  skinPhototype: SkinPhototype | null;
  familyHistorySkinCancer: boolean;
  familyHistoryRelation: string | null;
  familyHistoryOtherCancer: boolean;
  familyHistoryOtherCancerRelation: string | null;
  hadSevereSunburns: boolean;
  frequentSunExposure: boolean;
  usesTanningBeds: boolean;
  manyMoles: boolean;
  atypicalMoles: boolean;
  veryFairSkin: boolean;
};
```

- [ ] **Step 2: Commit**

```bash
git add melanoma-detection-app/types/user-profile.ts
git commit -m "feat: add UserProfile type"
```

---

## Task 2: UserProfile repository

**Files:**
- Create: `melanoma-detection-app/db/user-profile.repository.ts`

- [ ] **Step 1: Create the repository**

```typescript
// melanoma-detection-app/db/user-profile.repository.ts
import { getDb } from "./database";
import { UserProfile } from "@/types/user-profile";

type UserProfileRow = {
  id: number;
  age: number | null;
  skin_phototype: string | null;
  family_history_skin_cancer: number;
  family_history_relation: string | null;
  family_history_other_cancer: number;
  family_history_other_cancer_relation: string | null;
  had_severe_sunburns: number;
  frequent_sun_exposure: number;
  uses_tanning_beds: number;
  many_moles: number;
  atypical_moles: number;
  very_fair_skin: number;
};

function rowToProfile(row: UserProfileRow): UserProfile {
  return {
    id: row.id,
    age: row.age,
    skinPhototype: (row.skin_phototype as UserProfile["skinPhototype"]) ?? null,
    familyHistorySkinCancer: row.family_history_skin_cancer === 1,
    familyHistoryRelation: row.family_history_relation,
    familyHistoryOtherCancer: row.family_history_other_cancer === 1,
    familyHistoryOtherCancerRelation: row.family_history_other_cancer_relation,
    hadSevereSunburns: row.had_severe_sunburns === 1,
    frequentSunExposure: row.frequent_sun_exposure === 1,
    usesTanningBeds: row.uses_tanning_beds === 1,
    manyMoles: row.many_moles === 1,
    atypicalMoles: row.atypical_moles === 1,
    veryFairSkin: row.very_fair_skin === 1,
  };
}

export async function upsertUserProfile(profile: UserProfile): Promise<void> {
  const db = getDb();
  await db.runAsync(
    `INSERT OR REPLACE INTO user_profile (
      id, age, skin_phototype,
      family_history_skin_cancer, family_history_relation,
      family_history_other_cancer, family_history_other_cancer_relation,
      had_severe_sunburns, frequent_sun_exposure, uses_tanning_beds,
      many_moles, atypical_moles, very_fair_skin
    ) VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
    [
      profile.age,
      profile.skinPhototype,
      profile.familyHistorySkinCancer ? 1 : 0,
      profile.familyHistoryRelation ?? null,
      profile.familyHistoryOtherCancer ? 1 : 0,
      profile.familyHistoryOtherCancerRelation ?? null,
      profile.hadSevereSunburns ? 1 : 0,
      profile.frequentSunExposure ? 1 : 0,
      profile.usesTanningBeds ? 1 : 0,
      profile.manyMoles ? 1 : 0,
      profile.atypicalMoles ? 1 : 0,
      profile.veryFairSkin ? 1 : 0,
    ]
  );
}

export async function getUserProfile(): Promise<UserProfile | null> {
  const db = getDb();
  const row = await db.getFirstAsync<UserProfileRow>(
    `SELECT * FROM user_profile WHERE id = 1`
  );
  return row ? rowToProfile(row) : null;
}

export async function hasUserProfile(): Promise<boolean> {
  const profile = await getUserProfile();
  return profile !== null;
}
```

- [ ] **Step 2: Commit**

```bash
git add melanoma-detection-app/db/user-profile.repository.ts
git commit -m "feat: add user-profile repository"
```

---

## Task 3: Onboarding routing — disclaimer + _layout.tsx

**Files:**
- Modify: `melanoma-detection-app/app/_layout.tsx`
- Modify: `melanoma-detection-app/app/disclaimer.tsx`

Flow: disclaimer is ALWAYS shown first. After the user accepts and taps "Enter app", disclaimer checks AsyncStorage. If onboarding not done → go to `/onboarding`. If done → go to `/(tabs)`.

- [ ] **Step 1: Install AsyncStorage if not present**

```bash
cd melanoma-detection-app && npx expo install @react-native-async-storage/async-storage
```

Expected: package installed, no error.

- [ ] **Step 2: Update _layout.tsx — register onboarding route**

Replace the existing file with:

```typescript
// melanoma-detection-app/app/_layout.tsx
import React, { useEffect } from "react";
import { Stack } from "expo-router";
import { QueryClientProvider } from "@tanstack/react-query";
import { queryClient } from "@/lib/queryClient";
import { colors } from "@/theme/colors";
import { initDatabase } from "@/db/database";

export default function RootLayout() {
  useEffect(() => {
    void initDatabase();
  }, []);

  return (
    <QueryClientProvider client={queryClient}>
      <Stack
        initialRouteName="disclaimer"
        screenOptions={{
          headerShown: false,
          contentStyle: { backgroundColor: colors.bg },
        }}
      >
        <Stack.Screen name="disclaimer" />
        <Stack.Screen name="onboarding" />
        <Stack.Screen name="(tabs)" />
        <Stack.Screen name="lesions" />
        <Stack.Screen name="add-lesion" />
        <Stack.Screen name="lesion-details" />
        <Stack.Screen name="observation-details" />
      </Stack>
    </QueryClientProvider>
  );
}
```

- [ ] **Step 3: Update disclaimer.tsx — check onboarding after accept**

Replace the `onPress` handler of the "Enter app" Button in `disclaimer.tsx`:

```typescript
// Add import at the top of disclaimer.tsx
import AsyncStorage from "@react-native-async-storage/async-storage";
```

Replace the Button's `onPress` prop:

```typescript
onPress={async () => {
  const done = await AsyncStorage.getItem("onboarding_completed");
  if (done) {
    router.replace("/(tabs)");
  } else {
    router.replace("/onboarding");
  }
}}
```

- [ ] **Step 4: Commit**

```bash
git add melanoma-detection-app/app/_layout.tsx melanoma-detection-app/app/disclaimer.tsx
git commit -m "feat: add onboarding routing — disclaimer checks AsyncStorage before navigating"
```

---

## Task 4: ProfileWizard component

**Files:**
- Create: `melanoma-detection-app/components/ProfileWizard.tsx`

This is the shared 4-step form. Accepts `mode: "onboarding" | "edit"` and `onComplete: () => void`. All state is local. DB write happens only on final step submission.

- [ ] **Step 1: Create ProfileWizard.tsx**

```typescript
// melanoma-detection-app/components/ProfileWizard.tsx
import React, { useState } from "react";
import {
  View,
  Text,
  TextInput,
  ScrollView,
  Pressable,
  StyleSheet,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { Ionicons } from "@expo/vector-icons";
import { colors } from "@/theme/colors";
import Button from "@/components/Button";
import { UserProfile, SkinPhototype } from "@/types/user-profile";
import { upsertUserProfile } from "@/db/user-profile.repository";

type Props = {
  mode: "onboarding" | "edit";
  initialProfile?: UserProfile | null;
  onComplete: () => void;
};

const PHOTOTYPES: { value: SkinPhototype; label: string; description: string; color: string }[] = [
  { value: "I",   label: "Type I",   description: "Always burns, never tans. Pale/freckled.",      color: "#FDDCB5" },
  { value: "II",  label: "Type II",  description: "Usually burns, sometimes tans. Fair.",           color: "#F5C18C" },
  { value: "III", label: "Type III", description: "Sometimes burns, always tans. Light brown.",     color: "#D4956A" },
  { value: "IV",  label: "Type IV",  description: "Rarely burns, always tans. Olive/brown.",        color: "#A0693A" },
  { value: "V",   label: "Type V",   description: "Very rarely burns. Dark brown.",                 color: "#6B3F1F" },
  { value: "VI",  label: "Type VI",  description: "Never burns. Deeply pigmented.",                  color: "#3B1F0A" },
];

const TOTAL_STEPS = 4;

function ProgressBar({ step }: { step: number }) {
  return (
    <View style={styles.progressContainer}>
      {Array.from({ length: TOTAL_STEPS }).map((_, i) => (
        <View
          key={i}
          style={[
            styles.progressSegment,
            i < step && styles.progressSegmentDone,
          ]}
        />
      ))}
    </View>
  );
}

function Toggle({
  label,
  value,
  onChange,
}: {
  label: string;
  value: boolean;
  onChange: (v: boolean) => void;
}) {
  return (
    <Pressable
      style={[styles.toggle, value && styles.toggleActive]}
      onPress={() => onChange(!value)}
    >
      <Text style={[styles.toggleLabel, value && styles.toggleLabelActive]}>
        {label}
      </Text>
      <View style={[styles.toggleBox, value && styles.toggleBoxActive]}>
        {value && <Ionicons name="checkmark" size={13} color="#04090f" />}
      </View>
    </Pressable>
  );
}

export default function ProfileWizard({ mode, initialProfile, onComplete }: Props) {
  const [step, setStep] = useState(1);
  const [saving, setSaving] = useState(false);

  const [age, setAge] = useState(initialProfile?.age?.toString() ?? "");
  const [phototype, setPhototype] = useState<SkinPhototype | null>(
    initialProfile?.skinPhototype ?? null
  );
  const [familySkin, setFamilySkin] = useState(
    initialProfile?.familyHistorySkinCancer ?? false
  );
  const [familySkinWho, setFamilySkinWho] = useState(
    initialProfile?.familyHistoryRelation ?? ""
  );
  const [familyOther, setFamilyOther] = useState(
    initialProfile?.familyHistoryOtherCancer ?? false
  );
  const [familyOtherWho, setFamilyOtherWho] = useState(
    initialProfile?.familyHistoryOtherCancerRelation ?? ""
  );
  const [sunburns, setSunburns] = useState(
    initialProfile?.hadSevereSunburns ?? false
  );
  const [sunExposure, setSunExposure] = useState(
    initialProfile?.frequentSunExposure ?? false
  );
  const [tanning, setTanning] = useState(
    initialProfile?.usesTanningBeds ?? false
  );
  const [manyMoles, setManyMoles] = useState(initialProfile?.manyMoles ?? false);
  const [atypicalMoles, setAtypicalMoles] = useState(
    initialProfile?.atypicalMoles ?? false
  );
  const [fairSkin, setFairSkin] = useState(initialProfile?.veryFairSkin ?? false);

  async function handleFinish() {
    setSaving(true);
    const profile: UserProfile = {
      age: age ? parseInt(age, 10) : null,
      skinPhototype: phototype,
      familyHistorySkinCancer: familySkin,
      familyHistoryRelation: familySkin ? familySkinWho || null : null,
      familyHistoryOtherCancer: familyOther,
      familyHistoryOtherCancerRelation: familyOther ? familyOtherWho || null : null,
      hadSevereSunburns: sunburns,
      frequentSunExposure: sunExposure,
      usesTanningBeds: tanning,
      manyMoles,
      atypicalMoles,
      veryFairSkin: fairSkin,
    };
    await upsertUserProfile(profile);
    setSaving(false);
    onComplete();
  }

  const stepTitles = [
    "Basic info",
    "Family history",
    "Lifestyle",
    "Skin appearance",
  ];

  return (
    <SafeAreaView style={styles.safe}>
      <View style={styles.header}>
        {mode === "edit" && step === 1 ? null : (
          step > 1 ? (
            <Pressable onPress={() => setStep((s) => s - 1)} style={styles.back}>
              <Ionicons name="chevron-back" size={20} color={colors.text} />
            </Pressable>
          ) : <View style={styles.back} />
        )}
        <View style={styles.headerMeta}>
          <Text style={styles.stepLabel}>
            Step {step} of {TOTAL_STEPS}
          </Text>
          <Text style={styles.stepTitle}>{stepTitles[step - 1]}</Text>
        </View>
      </View>

      <ProgressBar step={step} />

      <ScrollView
        style={styles.scroll}
        contentContainerStyle={styles.scrollContent}
        showsVerticalScrollIndicator={false}
      >
        {step === 1 && (
          <View style={styles.fields}>
            <Text style={styles.fieldLabel}>Your age</Text>
            <TextInput
              style={styles.input}
              value={age}
              onChangeText={setAge}
              keyboardType="number-pad"
              placeholder="e.g. 34"
              placeholderTextColor={colors.muted}
              maxLength={3}
            />

            <Text style={[styles.fieldLabel, { marginTop: 20 }]}>
              Skin phototype (Fitzpatrick scale)
            </Text>
            <Text style={styles.fieldHint}>
              How does your skin react to sun exposure?
            </Text>
            {PHOTOTYPES.map((pt) => (
              <Pressable
                key={pt.value}
                style={[
                  styles.phototypeRow,
                  phototype === pt.value && styles.phototypeRowActive,
                ]}
                onPress={() => setPhototype(pt.value)}
              >
                <View
                  style={[styles.phototypeColor, { backgroundColor: pt.color }]}
                />
                <View style={styles.phototypeText}>
                  <Text style={styles.phototypeLabel}>{pt.label}</Text>
                  <Text style={styles.phototypeDesc}>{pt.description}</Text>
                </View>
                {phototype === pt.value && (
                  <Ionicons name="checkmark-circle" size={20} color={colors.primary} />
                )}
              </Pressable>
            ))}
          </View>
        )}

        {step === 2 && (
          <View style={styles.fields}>
            <Toggle
              label="Family history of skin cancer"
              value={familySkin}
              onChange={setFamilySkin}
            />
            {familySkin && (
              <>
                <Text style={[styles.fieldLabel, { marginTop: 12 }]}>
                  Who? (optional)
                </Text>
                <TextInput
                  style={styles.input}
                  value={familySkinWho}
                  onChangeText={setFamilySkinWho}
                  placeholder="e.g. mother, grandfather"
                  placeholderTextColor={colors.muted}
                />
              </>
            )}

            <View style={{ height: 16 }} />

            <Toggle
              label="Family history of other cancers"
              value={familyOther}
              onChange={setFamilyOther}
            />
            {familyOther && (
              <>
                <Text style={[styles.fieldLabel, { marginTop: 12 }]}>
                  Who? (optional)
                </Text>
                <TextInput
                  style={styles.input}
                  value={familyOtherWho}
                  onChangeText={setFamilyOtherWho}
                  placeholder="e.g. uncle"
                  placeholderTextColor={colors.muted}
                />
              </>
            )}
          </View>
        )}

        {step === 3 && (
          <View style={styles.fields}>
            <Toggle
              label="Severe sunburns in the past"
              value={sunburns}
              onChange={setSunburns}
            />
            <View style={{ height: 12 }} />
            <Toggle
              label="Frequent sun exposure"
              value={sunExposure}
              onChange={setSunExposure}
            />
            <View style={{ height: 12 }} />
            <Toggle
              label="Tanning bed use"
              value={tanning}
              onChange={setTanning}
            />
          </View>
        )}

        {step === 4 && (
          <View style={styles.fields}>
            <Toggle
              label="Many moles (more than 50)"
              value={manyMoles}
              onChange={setManyMoles}
            />
            <View style={{ height: 12 }} />
            <Toggle
              label="Atypical moles (irregular or large)"
              value={atypicalMoles}
              onChange={setAtypicalMoles}
            />
            <View style={{ height: 12 }} />
            <Toggle
              label="Very fair skin"
              value={fairSkin}
              onChange={setFairSkin}
            />
          </View>
        )}
      </ScrollView>

      <View style={styles.footer}>
        {step < TOTAL_STEPS ? (
          <Button title="Next" onPress={() => setStep((s) => s + 1)} />
        ) : (
          <Button
            title={saving ? "Saving…" : mode === "onboarding" ? "Enter app" : "Save"}
            disabled={saving}
            onPress={handleFinish}
          />
        )}
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: colors.bg },
  header: {
    flexDirection: "row",
    alignItems: "center",
    paddingHorizontal: 20,
    paddingTop: 12,
    paddingBottom: 8,
    gap: 12,
  },
  back: { width: 32, height: 32, alignItems: "center", justifyContent: "center" },
  headerMeta: { flex: 1 },
  stepLabel: { color: colors.muted, fontSize: 12, fontWeight: "600", letterSpacing: 0.4 },
  stepTitle: { color: colors.text, fontSize: 22, fontWeight: "800", letterSpacing: -0.5, marginTop: 2 },

  progressContainer: {
    flexDirection: "row",
    gap: 4,
    paddingHorizontal: 20,
    paddingBottom: 20,
  },
  progressSegment: {
    flex: 1,
    height: 4,
    borderRadius: 2,
    backgroundColor: colors.border,
  },
  progressSegmentDone: { backgroundColor: colors.primary },

  scroll: { flex: 1 },
  scrollContent: { paddingHorizontal: 20, paddingBottom: 24 },

  fields: { gap: 4 },
  fieldLabel: { color: colors.text, fontSize: 14, fontWeight: "600", marginBottom: 8 },
  fieldHint: { color: colors.subtext, fontSize: 13, marginBottom: 10, marginTop: -4 },

  input: {
    backgroundColor: colors.card,
    borderWidth: 1,
    borderColor: colors.border,
    borderRadius: 12,
    padding: 14,
    color: colors.text,
    fontSize: 15,
  },

  phototypeRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 12,
    padding: 12,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: colors.border,
    backgroundColor: colors.card,
    marginBottom: 6,
  },
  phototypeRowActive: {
    borderColor: colors.primaryBorder,
    backgroundColor: colors.primaryBg,
  },
  phototypeColor: { width: 28, height: 28, borderRadius: 8, flexShrink: 0 },
  phototypeText: { flex: 1 },
  phototypeLabel: { color: colors.text, fontSize: 14, fontWeight: "600" },
  phototypeDesc: { color: colors.subtext, fontSize: 12, marginTop: 2 },

  toggle: {
    flexDirection: "row",
    alignItems: "center",
    padding: 16,
    borderRadius: 14,
    borderWidth: 1,
    borderColor: colors.border,
    backgroundColor: colors.card,
    gap: 12,
  },
  toggleActive: {
    borderColor: colors.primaryBorder,
    backgroundColor: colors.primaryBg,
  },
  toggleLabel: { flex: 1, color: colors.text, fontSize: 14, fontWeight: "600" },
  toggleLabelActive: { color: colors.primary },
  toggleBox: {
    width: 24,
    height: 24,
    borderRadius: 7,
    borderWidth: 1.5,
    borderColor: colors.borderMid,
    backgroundColor: "rgba(255,255,255,0.04)",
    alignItems: "center",
    justifyContent: "center",
    flexShrink: 0,
  },
  toggleBoxActive: { backgroundColor: colors.primary, borderColor: colors.primary },

  footer: { padding: 20, paddingBottom: 8 },
});
```

- [ ] **Step 2: Commit**

```bash
git add melanoma-detection-app/components/ProfileWizard.tsx
git commit -m "feat: add ProfileWizard 4-step component"
```

---

## Task 5: Onboarding screen

**Files:**
- Create: `melanoma-detection-app/app/onboarding.tsx`

- [ ] **Step 1: Create onboarding screen**

```typescript
// melanoma-detection-app/app/onboarding.tsx
import AsyncStorage from "@react-native-async-storage/async-storage";
import { router } from "expo-router";
import ProfileWizard from "@/components/ProfileWizard";

export default function Onboarding() {
  async function handleComplete() {
    await AsyncStorage.setItem("onboarding_completed", "true");
    router.replace("/(tabs)");
  }

  return <ProfileWizard mode="onboarding" onComplete={handleComplete} />;
}
```

- [ ] **Step 2: Commit**

```bash
git add melanoma-detection-app/app/onboarding.tsx
git commit -m "feat: add onboarding screen"
```

- [ ] **Step 3: Manual smoke test**

  1. Clear app data / reinstall
  2. Launch app → should land on disclaimer
  3. Accept disclaimer → should redirect to onboarding (step 1 of 4)
  4. Navigate all 4 steps → tap "Enter app" → should land on main tabs
  5. Restart app → should NOT show onboarding again

---

## Task 6: Python API — risk scoring (TDD)

**Files:**
- Create: `melanoma-detection-ai/api/test_risk_scoring.py`
- Modify: `melanoma-detection-ai/api/main.py`

- [ ] **Step 1: Extract compute_clinical_risk to main.py**

Add below the FastAPI imports in `main.py`, before the model loading:

```python
from pydantic import BaseModel

class RiskProfile(BaseModel):
    age: int | None = None
    skin_phototype: str | None = None
    family_history_skin_cancer: bool = False
    family_history_other_cancer: bool = False
    had_severe_sunburns: bool = False
    frequent_sun_exposure: bool = False
    uses_tanning_beds: bool = False
    many_moles: bool = False
    atypical_moles: bool = False
    very_fair_skin: bool = False


RISK_WEIGHTS = {
    "family_history_skin_cancer": 3,
    "atypical_moles": 3,
    "family_history_other_cancer": 2,
    "many_moles": 2,
    "had_severe_sunburns": 1,
    "uses_tanning_beds": 1,
    "frequent_sun_exposure": 1,
    "very_fair_skin": 1,
}

PHOTOTYPE_LOW_RISK = {"I", "II"}


def compute_clinical_risk(profile: RiskProfile) -> tuple[str, float]:
    """Returns (clinical_risk_level, threshold)."""
    score = 0

    for field, weight in RISK_WEIGHTS.items():
        if getattr(profile, field, False):
            score += weight

    if profile.skin_phototype in PHOTOTYPE_LOW_RISK:
        score += 2

    if profile.age is not None and profile.age > 50:
        score += 1

    if score <= 2:
        return "low", 0.50
    elif score <= 5:
        return "medium", 0.42
    else:
        return "high", 0.35
```

- [ ] **Step 2: Write failing tests**

```python
# melanoma-detection-ai/api/test_risk_scoring.py
import pytest
from main import RiskProfile, compute_clinical_risk


def test_no_risk_factors_returns_low():
    profile = RiskProfile()
    level, threshold = compute_clinical_risk(profile)
    assert level == "low"
    assert threshold == 0.50


def test_family_history_skin_cancer_alone_is_medium():
    profile = RiskProfile(family_history_skin_cancer=True)
    level, threshold = compute_clinical_risk(profile)
    assert level == "medium"
    assert threshold == 0.42


def test_family_history_and_atypical_moles_is_high():
    profile = RiskProfile(family_history_skin_cancer=True, atypical_moles=True)
    level, threshold = compute_clinical_risk(profile)
    assert level == "high"
    assert threshold == 0.35


def test_phototype_I_adds_score():
    profile = RiskProfile(skin_phototype="I")
    level, _ = compute_clinical_risk(profile)
    assert level == "medium"


def test_phototype_III_no_extra_score():
    profile = RiskProfile(skin_phototype="III")
    level, threshold = compute_clinical_risk(profile)
    assert level == "low"
    assert threshold == 0.50


def test_age_over_50_adds_score():
    profile = RiskProfile(age=55, family_history_other_cancer=True)
    level, _ = compute_clinical_risk(profile)
    assert level == "medium"


def test_all_factors_high():
    profile = RiskProfile(
        family_history_skin_cancer=True,
        atypical_moles=True,
        many_moles=True,
        skin_phototype="I",
        age=60,
    )
    level, threshold = compute_clinical_risk(profile)
    assert level == "high"
    assert threshold == 0.35
```

- [ ] **Step 3: Run tests — expect FAIL (function not yet in place)**

```bash
cd melanoma-detection-ai && .venv/bin/pytest api/test_risk_scoring.py -v
```

Expected: ImportError or NameError (compute_clinical_risk not defined yet — that's fine, it confirms test isolation).

- [ ] **Step 4: Add compute_clinical_risk to main.py (from Step 1 above) and run tests**

```bash
.venv/bin/pytest api/test_risk_scoring.py -v
```

Expected: all 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add melanoma-detection-ai/api/main.py melanoma-detection-ai/api/test_risk_scoring.py
git commit -m "feat: add RiskProfile model and compute_clinical_risk with tests"
```

---

## Task 7: Python API — extend /predict endpoint

**Files:**
- Modify: `melanoma-detection-ai/api/main.py`

- [ ] **Step 1: Update /predict to accept optional risk_profile**

Replace the existing `@app.post("/predict")` function with:

```python
import json

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    risk_profile: str | None = Form(default=None),
):
    if file.content_type not in ("image/jpeg", "image/png", "image/webp"):
        raise HTTPException(status_code=400, detail="Wgraj obraz JPG/PNG/WEBP.")

    data = await file.read()
    if len(data) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Plik za duży (max 10 MB).")

    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Niepoprawny plik obrazu.")

    x = tf(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logit = model(x).squeeze(1)
        prob = torch.sigmoid(logit).item()

    clinical_risk_level: str | None = None
    threshold = THRESHOLD

    if risk_profile:
        try:
            profile_data = json.loads(risk_profile)
            profile = RiskProfile(**profile_data)
            clinical_risk_level, threshold = compute_clinical_risk(profile)
        except Exception:
            pass  # malformed profile → fall back to default threshold

    label: Literal["low_risk", "high_risk"] = "high_risk" if prob >= threshold else "low_risk"

    return {
        "probability": prob,
        "threshold": threshold,
        "clinical_risk_level": clinical_risk_level,
        "label": label,
        "disclaimer": "This is not a medical diagnosis. Consult a dermatologist.",
    }
```

Also add `Form` to the fastapi import line:
```python
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
```

- [ ] **Step 2: Run existing tests still pass**

```bash
cd melanoma-detection-ai && .venv/bin/pytest api/test_risk_scoring.py -v
```

Expected: all 7 PASS.

- [ ] **Step 3: Commit**

```bash
git add melanoma-detection-ai/api/main.py
git commit -m "feat: extend /predict to accept and apply risk_profile"
```

---

## Task 8: Update TypeScript predict client

**Files:**
- Modify: `melanoma-detection-app/lib/api/predict.ts`

- [ ] **Step 1: Update types and send risk_profile**

Replace the file contents with:

```typescript
// melanoma-detection-app/lib/api/predict.ts
import Constants from "expo-constants";
import { Platform } from "react-native";
import { getUserProfile } from "@/db/user-profile.repository";
import { UserProfile } from "@/types/user-profile";

type PredictInput = {
  uri: string;
};

type PredictApiResponse = {
  probability: number;
  threshold: number;
  clinical_risk_level: "low" | "medium" | "high" | null;
  label: "low_risk" | "high_risk";
  disclaimer: string;
};

export type PredictResult = {
  isMelanoma: boolean;
  score: number;
  clinicalRiskLevel: "low" | "medium" | "high" | null;
  thresholdUsed: number;
  raw: PredictApiResponse;
};

function resolveApiUrl(): string {
  const extra = (Constants.expoConfig?.extra || Constants.manifest?.extra || {}) as { apiUrl?: string };
  const envUrl = process.env.EXPO_PUBLIC_API_URL;
  const rawUrl = envUrl || extra.apiUrl;

  if (!rawUrl) throw new Error("Missing EXPO_PUBLIC_API_URL (.env) or expo extra.apiUrl");

  const normalized = rawUrl.endsWith("/") ? rawUrl.slice(0, -1) : rawUrl;

  if (Platform.OS === "android" && normalized.includes("localhost")) {
    return normalized.replace("localhost", "10.0.2.2");
  }

  return normalized;
}

function profileToApiPayload(profile: UserProfile): Record<string, unknown> {
  return {
    age: profile.age,
    skin_phototype: profile.skinPhototype,
    family_history_skin_cancer: profile.familyHistorySkinCancer,
    family_history_other_cancer: profile.familyHistoryOtherCancer,
    had_severe_sunburns: profile.hadSevereSunburns,
    frequent_sun_exposure: profile.frequentSunExposure,
    uses_tanning_beds: profile.usesTanningBeds,
    many_moles: profile.manyMoles,
    atypical_moles: profile.atypicalMoles,
    very_fair_skin: profile.veryFairSkin,
  };
}

const API_URL = resolveApiUrl();

export async function predict({ uri }: PredictInput): Promise<PredictResult> {
  const form = new FormData();

  form.append("file", {
    uri,
    name: "image.jpg",
    type: "image/jpeg",
  } as any);

  const profile = await getUserProfile();
  if (profile) {
    form.append("risk_profile", JSON.stringify(profileToApiPayload(profile)));
  }

  const res = await fetch(`${API_URL}/predict`, {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    const errorText = await res.text().catch(() => "");
    let parsed: any;
    try {
      parsed = errorText ? JSON.parse(errorText) : null;
    } catch {
      parsed = null;
    }
    throw new Error(parsed?.detail || errorText || `Request failed (${res.status})`);
  }

  const data = (await res.json()) as PredictApiResponse;

  return {
    isMelanoma: data.label === "high_risk",
    score: data.probability,
    clinicalRiskLevel: data.clinical_risk_level,
    thresholdUsed: data.threshold,
    raw: data,
  };
}
```

- [ ] **Step 2: Update upload.tsx to pass new result fields to result screen**

In `melanoma-detection-app/app/(tabs)/upload.tsx`, update the `router.replace` call inside `onAnalyze`:

```typescript
router.replace({
  pathname: "/result",
  params: {
    isMelanoma: String(res.isMelanoma),
    score: String(res.score),
    clinicalRiskLevel: res.clinicalRiskLevel ?? "",
    thresholdUsed: String(res.thresholdUsed),
  },
});
```

- [ ] **Step 3: Commit**

```bash
git add melanoma-detection-app/lib/api/predict.ts melanoma-detection-app/app/(tabs)/upload.tsx
git commit -m "feat: send risk_profile to API and surface clinical_risk_level in result"
```

---

## Task 9: Update result screen

**Files:**
- Modify: `melanoma-detection-app/app/result.tsx`

- [ ] **Step 1: Replace result.tsx**

```typescript
// melanoma-detection-app/app/result.tsx
import { useLocalSearchParams, useRouter } from "expo-router";
import { StyleSheet, Text, View } from "react-native";
import Screen from "@/components/Screen";
import { colors } from "@/theme/colors";
import Button from "@/components/Button";

const RISK_BADGE: Record<string, { label: string; color: string; bg: string }> = {
  low:    { label: "Low clinical risk",    color: colors.success, bg: colors.successBg },
  medium: { label: "Medium clinical risk", color: colors.warning, bg: "rgba(251,191,36,0.10)" },
  high:   { label: "High clinical risk",   color: colors.danger,  bg: colors.dangerBg },
};

export default function Result() {
  const { isMelanoma, score, clinicalRiskLevel, thresholdUsed } =
    useLocalSearchParams<{
      isMelanoma: string;
      score: string;
      clinicalRiskLevel: string;
      thresholdUsed: string;
    }>();

  const melanoma = isMelanoma === "true";
  const color = melanoma ? colors.danger : colors.success;
  const router = useRouter();
  const badge = clinicalRiskLevel ? RISK_BADGE[clinicalRiskLevel] : null;
  const threshold = thresholdUsed ? parseFloat(thresholdUsed) : 0.5;
  const thresholdAdjusted = threshold !== 0.5;

  return (
    <Screen>
      <View style={styles.center}>
        <Text style={[styles.result, { color }]}>
          {melanoma ? "Suspicious (possible melanoma)" : "Likely benign"}
        </Text>
        <Text style={styles.score}>
          Confidence: {(Number(score) * 100).toFixed(1)}%
        </Text>

        {badge && (
          <View style={[styles.badge, { backgroundColor: badge.bg, borderColor: badge.color + "44" }]}>
            <Text style={[styles.badgeText, { color: badge.color }]}>{badge.label}</Text>
          </View>
        )}

        {thresholdAdjusted && (
          <Text style={styles.thresholdNote}>
            Detection threshold adjusted to {(threshold * 100).toFixed(0)}% based on your risk profile.
          </Text>
        )}

        <Text style={styles.disclaimer}>Not a medical diagnosis.</Text>
        <Button
          title="Go Home"
          onPress={() => router.replace("/")}
          style={{ marginTop: 24 }}
        />
      </View>
    </Screen>
  );
}

const styles = StyleSheet.create({
  center: { flex: 1, alignItems: "center", justifyContent: "center", padding: 16, gap: 8 },
  result: { fontSize: 22, fontWeight: "700", textAlign: "center" },
  score: { color: colors.text },
  badge: {
    paddingHorizontal: 14,
    paddingVertical: 6,
    borderRadius: 20,
    borderWidth: 1,
    marginTop: 8,
  },
  badgeText: { fontSize: 13, fontWeight: "600" },
  thresholdNote: {
    color: colors.subtext,
    fontSize: 13,
    textAlign: "center",
    marginTop: 4,
    paddingHorizontal: 16,
  },
  disclaimer: { marginTop: 16, color: colors.subtext, textAlign: "center" },
});
```

- [ ] **Step 2: Commit**

```bash
git add melanoma-detection-app/app/result.tsx
git commit -m "feat: show clinical_risk_level badge and threshold info on result screen"
```

---

## Task 10: Update profile tab

**Files:**
- Modify: `melanoma-detection-app/app/(tabs)/profile.tsx`

- [ ] **Step 1: Update profile.tsx to show risk profile section and Edit button**

Add to the existing `profile.tsx`. The key additions are: fetch profile on focus, render risk badge, list active risk factors, add "Edit profile" menu item that navigates to `/onboarding` in edit mode.

First, add the edit route to `_layout.tsx` Stack (it reuses the onboarding screen with edit mode — we need a separate route):

Create `melanoma-detection-app/app/edit-profile.tsx`:

```typescript
// melanoma-detection-app/app/edit-profile.tsx
import { router } from "expo-router";
import { useEffect, useState } from "react";
import ProfileWizard from "@/components/ProfileWizard";
import { getUserProfile } from "@/db/user-profile.repository";
import { UserProfile } from "@/types/user-profile";

export default function EditProfile() {
  const [profile, setProfile] = useState<UserProfile | null>(null);
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    getUserProfile().then((p) => {
      setProfile(p);
      setLoaded(true);
    });
  }, []);

  if (!loaded) return null;

  return (
    <ProfileWizard
      mode="edit"
      initialProfile={profile}
      onComplete={() => router.back()}
    />
  );
}
```

- [ ] **Step 2: Register edit-profile route in _layout.tsx**

Add inside the `<Stack>` in `_layout.tsx`:

```typescript
<Stack.Screen name="edit-profile" />
```

- [ ] **Step 3: Update profile.tsx**

Replace the file with:

```typescript
// melanoma-detection-app/app/(tabs)/profile.tsx
import { useCallback, useState } from "react";
import { Pressable, ScrollView, StyleSheet, Text, View } from "react-native";
import { useRouter } from "expo-router";
import { useFocusEffect } from "@react-navigation/native";
import { SafeAreaView } from "react-native-safe-area-context";
import { Ionicons } from "@expo/vector-icons";
import { colors } from "@/theme/colors";
import { getObservationsCount } from "@/db/observations.repository";
import { getLesionsCount } from "@/db/lesions.repository";
import { getUserProfile } from "@/db/user-profile.repository";
import { UserProfile, ClinicalRiskLevel } from "@/types/user-profile";

type MenuItem = {
  icon: React.ComponentProps<typeof Ionicons>["name"];
  label: string;
  sub: string;
  onPress: () => void;
  danger?: boolean;
};

const RISK_COLORS: Record<ClinicalRiskLevel, { color: string; bg: string }> = {
  low:    { color: colors.success, bg: colors.successBg },
  medium: { color: colors.warning, bg: "rgba(251,191,36,0.10)" },
  high:   { color: colors.danger,  bg: colors.dangerBg },
};

function computeLocalRiskLevel(profile: UserProfile): ClinicalRiskLevel {
  let score = 0;
  if (profile.familyHistorySkinCancer) score += 3;
  if (profile.atypicalMoles) score += 3;
  if (profile.familyHistoryOtherCancer) score += 2;
  if (profile.manyMoles) score += 2;
  if (profile.skinPhototype === "I" || profile.skinPhototype === "II") score += 2;
  if (profile.hadSevereSunburns) score += 1;
  if (profile.usesTanningBeds) score += 1;
  if (profile.frequentSunExposure) score += 1;
  if (profile.veryFairSkin) score += 1;
  if (profile.age !== null && profile.age !== undefined && profile.age > 50) score += 1;
  if (score <= 2) return "low";
  if (score <= 5) return "medium";
  return "high";
}

const RISK_FACTOR_LABELS: { key: keyof UserProfile; label: string }[] = [
  { key: "familyHistorySkinCancer",  label: "Family history of skin cancer" },
  { key: "atypicalMoles",            label: "Atypical moles" },
  { key: "familyHistoryOtherCancer", label: "Family history of other cancers" },
  { key: "manyMoles",                label: "Many moles (>50)" },
  { key: "hadSevereSunburns",        label: "Severe sunburns in the past" },
  { key: "usesTanningBeds",          label: "Tanning bed use" },
  { key: "frequentSunExposure",      label: "Frequent sun exposure" },
  { key: "veryFairSkin",             label: "Very fair skin" },
];

export default function ProfileScreen() {
  const router = useRouter();
  const [counts, setCounts] = useState({ observations: 0, lesions: 0 });
  const [profile, setProfile] = useState<UserProfile | null>(null);

  useFocusEffect(
    useCallback(() => {
      void Promise.all([
        getObservationsCount(),
        getLesionsCount(),
        getUserProfile(),
      ]).then(([observations, lesions, p]) => {
        setCounts({ observations, lesions });
        setProfile(p);
      });
    }, []),
  );

  const riskLevel = profile ? computeLocalRiskLevel(profile) : null;
  const riskStyle = riskLevel ? RISK_COLORS[riskLevel] : null;

  const activeFactors = profile
    ? RISK_FACTOR_LABELS.filter((f) => profile[f.key] === true)
    : [];

  const menu: MenuItem[] = [
    {
      icon: "body-outline",
      label: "My lesions",
      sub: "Manage tracked spots",
      onPress: () => router.push("/lesions"),
    },
    {
      icon: "create-outline",
      label: "Edit profile",
      sub: "Update your risk factors",
      onPress: () => router.push("/edit-profile"),
    },
  ];

  return (
    <SafeAreaView style={styles.safe}>
      <ScrollView contentContainerStyle={styles.container} showsVerticalScrollIndicator={false}>

        <View style={styles.avatar}>
          <Ionicons name="person" size={28} color={colors.primary} />
        </View>
        <Text style={styles.name}>Your profile</Text>
        <Text style={styles.nameSub}>Manage your health data</Text>

        <View style={styles.stats}>
          <View style={styles.stat}>
            <Text style={styles.statValue}>{counts.observations}</Text>
            <Text style={styles.statLabel}>Analyses</Text>
          </View>
          <View style={styles.statDivider} />
          <View style={styles.stat}>
            <Text style={styles.statValue}>{counts.lesions}</Text>
            <Text style={styles.statLabel}>Lesions</Text>
          </View>
        </View>

        {riskLevel && riskStyle && (
          <View style={styles.riskSection}>
            <Text style={styles.sectionTitle}>Risk Profile</Text>
            <View style={[styles.riskBadge, { backgroundColor: riskStyle.bg, borderColor: riskStyle.color + "44" }]}>
              <Text style={[styles.riskBadgeText, { color: riskStyle.color }]}>
                {riskLevel.charAt(0).toUpperCase() + riskLevel.slice(1)} clinical risk
              </Text>
            </View>

            {activeFactors.length > 0 && (
              <View style={styles.factorsList}>
                {activeFactors.map((f) => (
                  <View key={f.key as string} style={styles.factorRow}>
                    <Ionicons name="alert-circle-outline" size={14} color={colors.warning} />
                    <Text style={styles.factorLabel}>{f.label}</Text>
                  </View>
                ))}
              </View>
            )}

            {profile?.skinPhototype && (
              <Text style={styles.phototypeNote}>
                Skin phototype: Fitzpatrick {profile.skinPhototype}
              </Text>
            )}
          </View>
        )}

        <View style={styles.menu}>
          {menu.map((item, i) => (
            <Pressable
              key={item.label}
              style={[
                styles.menuItem,
                i < menu.length - 1 && styles.menuItemBorder,
              ]}
              onPress={item.onPress}
            >
              <View style={styles.menuIcon}>
                <Ionicons name={item.icon} size={18} color={colors.primary} />
              </View>
              <View style={styles.menuText}>
                <Text style={styles.menuLabel}>{item.label}</Text>
                <Text style={styles.menuSub}>{item.sub}</Text>
              </View>
              <Ionicons name="chevron-forward" size={16} color={colors.muted} />
            </Pressable>
          ))}
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: colors.bg },
  container: { padding: 20, paddingBottom: 40 },
  avatar: {
    width: 70,
    height: 70,
    borderRadius: 22,
    backgroundColor: colors.primaryBg,
    borderWidth: 1,
    borderColor: colors.primaryBorder,
    alignItems: "center",
    justifyContent: "center",
    marginBottom: 14,
    alignSelf: "flex-start",
  },
  name: { color: colors.text, fontSize: 28, fontWeight: "800", letterSpacing: -0.8, marginBottom: 4 },
  nameSub: { color: colors.subtext, fontSize: 14, marginBottom: 24 },
  stats: {
    backgroundColor: colors.card,
    borderRadius: 18,
    borderWidth: 1,
    borderColor: colors.border,
    flexDirection: "row",
    alignItems: "center",
    paddingVertical: 20,
    paddingHorizontal: 24,
    marginBottom: 18,
  },
  stat: { flex: 1, alignItems: "center" },
  statValue: { color: colors.text, fontSize: 32, fontWeight: "800", letterSpacing: -1, marginBottom: 4 },
  statLabel: { color: colors.muted, fontSize: 12, fontWeight: "600", letterSpacing: 0.3 },
  statDivider: { width: 1, height: 40, backgroundColor: colors.border },

  sectionTitle: { color: colors.text, fontSize: 16, fontWeight: "700", marginBottom: 10 },
  riskSection: {
    backgroundColor: colors.card,
    borderRadius: 18,
    borderWidth: 1,
    borderColor: colors.border,
    padding: 16,
    marginBottom: 18,
    gap: 10,
  },
  riskBadge: {
    alignSelf: "flex-start",
    paddingHorizontal: 14,
    paddingVertical: 6,
    borderRadius: 20,
    borderWidth: 1,
  },
  riskBadgeText: { fontSize: 13, fontWeight: "600" },
  factorsList: { gap: 6 },
  factorRow: { flexDirection: "row", alignItems: "center", gap: 8 },
  factorLabel: { color: colors.subtext, fontSize: 13 },
  phototypeNote: { color: colors.muted, fontSize: 12 },

  menu: {
    backgroundColor: colors.card,
    borderRadius: 18,
    borderWidth: 1,
    borderColor: colors.border,
    overflow: "hidden",
  },
  menuItem: { flexDirection: "row", alignItems: "center", padding: 16, gap: 14 },
  menuItemBorder: { borderBottomWidth: 1, borderBottomColor: colors.border },
  menuIcon: {
    width: 38,
    height: 38,
    borderRadius: 11,
    backgroundColor: colors.primaryBg,
    borderWidth: 1,
    borderColor: colors.primaryBorder,
    alignItems: "center",
    justifyContent: "center",
  },
  menuText: { flex: 1 },
  menuLabel: { color: colors.text, fontSize: 15, fontWeight: "600", marginBottom: 2 },
  menuSub: { color: colors.muted, fontSize: 12 },
});
```

- [ ] **Step 4: Commit**

```bash
git add melanoma-detection-app/app/(tabs)/profile.tsx melanoma-detection-app/app/edit-profile.tsx melanoma-detection-app/app/_layout.tsx
git commit -m "feat: show risk profile section in profile tab, add edit-profile route"
```

---

## Task 11: End-to-end manual test

- [ ] **Step 1: Start API server**

```bash
cd melanoma-detection-ai && docker-compose up
```

Or directly:
```bash
cd melanoma-detection-ai && .venv/bin/uvicorn api.main:app --reload --port 8000
```

- [ ] **Step 2: Start Expo app**

```bash
cd melanoma-detection-app && pnpm start
```

- [ ] **Step 3: Test onboarding flow**

1. Fresh install → disclaimer → onboarding wizard appears
2. Fill all 4 steps (set age=60, phototype=I, family_history_skin_cancer=true, atypical_moles=true)
3. Tap "Enter app" → main tabs load
4. Restart → onboarding does NOT show again

- [ ] **Step 4: Test analysis with risk profile**

1. Upload tab → pick image → Analyze
2. Result screen shows clinical_risk_level badge (should be "High" for the profile above)
3. Threshold note visible: "Detection threshold adjusted to 35%..."

- [ ] **Step 5: Test profile editing**

1. Profile tab → risk profile section visible with badge + factor list
2. Tap "Edit profile" → wizard opens pre-filled
3. Change a value → Save → profile tab updates

- [ ] **Step 6: Test API fallback**

1. Temporarily comment out profile fetch in predict.ts (pass no risk_profile)
2. Analyze → result shows no badge, no threshold note
3. Restore

- [ ] **Step 7: Commit if any fixes needed, then tag**

```bash
git tag v0.2.0-risk-profile
```