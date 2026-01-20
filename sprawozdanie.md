# Sprawozdanie – Melanoma Detection Core

## 1. Opis projektu

Przygotowano kompletny system wspierający ocenę zmian skórnych pod kątem czerniaka. Na rozwiązanie składają się:

- wytrenowany model EfficientNet-B0 w PyTorchu,
- backend FastAPI udostępniający REST API,
- aplikacja mobilna Expo/React Native, która prowadzi użytkownika przez proces wykonania zdjęcia i prezentuje wynik.

System ma charakter przesiewowy i edukacyjny – wskazuje, czy zmiana wymaga konsultacji dermatologicznej, lecz nie stanowi diagnozy medycznej.

---

## 2. Architektura rozwiązania

```
telefon / emulator → aplikacja Expo → FastAPI → EfficientNet-B0
```

1. Użytkownik wykonuje zdjęcie lub wybiera je z galerii.
2. Aplikacja wysyła żądanie `POST /predict` wraz z plikiem `multipart/form-data`.
3. FastAPI weryfikuje dane wejściowe, przygotowuje tensor i uruchamia inferencję.
4. Wynik (prawdopodobieństwo, etykieta, komunikat) trafia z powrotem do aplikacji i jest wyświetlany użytkownikowi.

Repozytoria:
- `melanoma-detection-ai` – model, pipeline i backend.
- `melanoma-detection-app` – aplikacja mobilna Expo Router.

---

## 3. Dane i przygotowanie zbioru

### Zbiór danych
- HAM10000 (Human Against Machine with 10000 training images): 10 015 obrazów dermatoskopowych i klinicznych.
- Etykiety `dx` zmapowano do klasy binarnej: `mel` (czerniak) versus pozostałe zmiany (łagodne).
- Zbiór jest niezbalansowany – przypadki czerniaka stanowią około 11% danych.

### Podział i preprocessing
- Stratyfikowany podział train/validation/test: 80% / 10% / 10%.
- Preprocessing: konwersja do RGB, zmiana rozmiaru do 224×224, konwersja do tensora, normalizacja według statystyk ImageNet.
- Testowano dodatkowe augmentacje imitujące zdjęcia z telefonów (rozmycie, zmiana oświetlenia, mocne kadrowanie). Spowodowały spadek AUC i swoistości, dlatego finalny model korzysta z klasycznych augmentacji.

---

## 4. Model i proces uczenia

### Architektura i trening
- EfficientNet-B0 inicjalizowany wagami ImageNet.
- Końcowa warstwa zastąpiona pojedynczym neuronem (logit → sigmoid).
- Funkcja straty: `BCEWithLogitsLoss` z `pos_weight` (kompensacja niezbalansowanych klas).
- Optymalizator: `AdamW`, scheduler: `ReduceLROnPlateau`.
- Zastosowano mechanizm early stopping na zbiorze walidacyjnym.
- Najlepszy checkpoint zapisano do `artifacts/efficientnet_b0_best.pt`.

### Metryki i próg
- Na zbiorze testowym uzyskano: AUC ≈ 0.94, Recall ≈ 0.86, Specificity ≈ 0.89.
- Analizowano progi 0.361 / 0.50 / 0.559. Ostatecznie przyjęto **0.50** jako kompromis między czułością a swoistością.

---

## 5. Backend: `melanoma-detection-ai`

### Model w API
- EfficientNet-B0 wczytywany podczas startu serwera i przełączony w tryb `eval`.
- Próg decyzyjny 0.50 definiuje etykiety `low_risk` / `high_risk`.

### Przetwarzanie obrazu
1. Wczytanie pliku przez PIL i konwersja do RGB.
2. Transformacje torchvision (224×224, `ToTensor`, normalizacja).

### Endpointy
- `GET /health` – status serwera (CPU/GPU), próg, podstawowe dane diagnostyczne.
- `POST /predict` – przyjmuje JPEG/PNG/WebP do 10 MB i zwraca m.in.:

```json
{
  "probability": 0.73,
  "threshold": 0.5,
  "label": "high_risk",
  "disclaimer": "This is not a medical diagnosis..."
}
```

Zwracany jest również komunikat ostrzegawczy. CORS dopuszcza połączenia z dowolnego hosta, co umożliwia testy na urządzeniach mobilnych w sieci LAN.

### Uruchomienie

```bash
cd melanoma-detection-ai
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Skrypt `scripts/dev.sh` automatyzuje tworzenie środowiska, instalację zależności i start Uvicorna (ustawia m.in. `OMP_NUM_THREADS=1` oraz `PYTORCH_ENABLE_MPS_FALLBACK=1`).

---

## 6. Aplikacja mobilna: `melanoma-detection-app`

### Technologie
- Expo Router (file-based routing),
- React Native 0.81 + TypeScript,
- Expo Image Picker (kamera i galeria),
- TanStack Query (`usePredictMutation`) do obsługi zapytań,
- QueryClient z ustawieniami dostosowanymi do aplikacji mobilnych (wyłączone refetch-on-focus).

### Struktura widoków
1. **Home** – opis aplikacji, wezwanie do działania i nawigacja do zakładki Upload.
2. **Upload** – obsługuje wybór/wykonanie zdjęcia, pokazuje podgląd, umożliwia wysłanie pliku do analizy i zarządza stanem oczekiwania.
3. **Instructions** – przedstawia krok po kroku sposób wykonania badania oraz wskazówki dotyczące jakości zdjęć.
4. **Result** – prezentuje wynik (`low_risk`/`high_risk`), procentowe prawdopodobieństwo, kolorystykę zgodną z wynikiem oraz komunikat ostrzegawczy.

### Logika interfejsu
- `ImagePickerActions` wyświetla dwa przyciski (galeria / aparat), prosi o odpowiednie uprawnienia i przekazuje URI wybranego zdjęcia.
- `Button` dezaktywuje się w trakcie żądania oraz gdy użytkownik nie wybrał jeszcze zdjęcia.
- `app/(tabs)/upload.tsx` kontroluje stan URI, reaguje na kliknięcie „Analyze” i prezentuje komunikaty o błędach (np. 400/500 z backendu).
- `app/result.tsx` odbiera parametry `isMelanoma` i `score` z routera, dobiera kolorystykę oraz wyświetla szczegółowe komunikaty.

Fragment obsługi wysyłki i błędów w widoku Upload:

```tsx
const onAnalyze = async () => {
  if (!uri) return Alert.alert("Please select a photo first.");
  try {
    const res = await mutateAsync({ uri });
    router.replace({
      pathname: "/result",
      params: {
        isMelanoma: String(res.isMelanoma),
        score: String(res.score),
      },
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unexpected error";
    Alert.alert("Prediction failed", message);
  }
};
```

### Integracja z API
`lib/api/predict.ts` wykonuje następujące kroki:
- Pobiera bazowy URL z `EXPO_PUBLIC_API_URL` (plik `.env`) lub z `app.json` (`expo.extra.apiUrl`).
- Normalizuje końcówkę adresu i w emulatorze Androida podmienia `localhost` na `10.0.2.2`.
- Tworzy `FormData` z obrazem i wysyła zapytanie `fetch` na `POST /predict`.
- Interpretuje błędy (JSON lub plaintext) i zwraca znormalizowane dane (`PredictResult`).

Przykładowy fragment JSX kontrolujący stan podglądu i blokadę przycisku:

```tsx
return (
  <Screen>
    <View style={styles.container}>
      <ImagePickerActions onPick={setUri} />
      {uri ? (
        <Image source={{ uri }} style={styles.preview} />
      ) : (
        <Text style={styles.tip}>No image selected</Text>
      )}
      <Button
        title="Analyze"
        onPress={onAnalyze}
        loading={isPending}
        disabled={!uri}
        style={{ marginTop: 16 }}
      />
      {isPending && <ActivityIndicator style={{ marginTop: 16 }} />}
    </View>
  </Screen>
);
```

### Konfiguracja środowiska
- `.env.example` zawiera wzorcowy wpis `EXPO_PUBLIC_API_URL=http://192.168.0.100:8000`.
- README opisuje kolejność działań: uruchom backend → skopiuj `.env.example` → `pnpm install` → `pnpm start`.
- ESLint (flat JS/TS/React) został dostosowany do środowiska Expo, aby wspierać importy assetów (`require`).

---

## 7. Kierunki rozwoju
- Historia badań i konta użytkowników (np. zapis wyników, porównanie zmian w czasie).
- Segmentacja zmiany lub wizualizacje typu Grad-CAM, aby pokazać, które fragmenty obrazu wpłynęły na wynik.
- Inferencja offline (TensorFlow Lite/CoreML) dla nowszych urządzeń.
- Integracja telemedyczna – przesyłanie zdjęć do lekarzy wraz z danymi pacjenta.
- Dynamiczne dostrajanie progu decyzyjnego na podstawie danych z użytkowania aplikacji.

---

## 8. Podsumowanie

Rozwiązanie łączy warstwę uczenia maszynowego, backend oraz aplikację mobilną w spójny przepływ danych. Dzięki czytelnej dokumentacji i skryptom uruchomieniowym uruchomienie projektu jest szybkie, a architektura pozostaje otwarta na kolejne rozszerzenia. Model utrzymuje wysokie metryki (AUC, recall), backend udostępnia stabilne API, natomiast aplikacja Expo zapewnia przyjazny interfejs użytkownika i obejmuje pełną ścieżkę od wykonania zdjęcia po interpretację wyniku.
