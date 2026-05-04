# Risk Profile Onboarding — Overview

**Goal:** Add mandatory 4-step onboarding collecting clinical risk factors. Factors stored in SQLite, sent to AI API on every analysis, used to dynamically adjust the detection threshold.

**Backend and frontend can be implemented in parallel. Integration tasks (F6–F8, I1) require both.**

---

## Plans

| Repo | File | Tasks |
|------|------|-------|
| `melanoma-detection-ai` | `docs/superpowers/plans/2026-05-04-risk-profile-onboarding.md` | B1, B2 |
| `melanoma-detection-app` | `docs/superpowers/plans/2026-05-04-risk-profile-onboarding.md` | F1–F8, I1 |

---

## Task order

```
B1 (risk scoring + tests)  ─┐
B2 (extend /predict)        ├─→ F6 (predict client) → F7 (result screen) → I1 (e2e test)
                            │
F1 (UserProfile type)      ─┤
F2 (repository)             │
F3 (routing)               ─┘
F4 (ProfileWizard)
F5 (onboarding + edit screens)    → F8 (profile tab)
```

---

## Design spec

`docs/superpowers/specs/2026-05-04-risk-profile-onboarding-design.md`
