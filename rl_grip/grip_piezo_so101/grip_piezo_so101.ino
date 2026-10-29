const int    PIN_PIEZO     = A0;
const float  VCC_MV        = 5000.0;   // reference voltage in mV
const int    ADC_MAX       = 1023;
const float  THRESHOLD_MV  = 1.0;      // detection threshold (mV above offset)
const int    WINDOW        = 20;       // number of samples for peak tracking
const int    DELAY_MS      = 5;        // sampling period (200 Hz)
const int    COOLDOWN_MS   = 10;       // pause after peak emission

// Global variables
float currentPeak = 0.0;
bool  inContact   = false;
unsigned long lastPeakTime = 0;

// Circular buffer for gradient
float buf[WINDOW];
int   bufIdx = 0;
bool  bufFull = false;

// Setup
void setup() {
  Serial.begin(9600);
  pinMode(PIN_PIEZO, INPUT);
  while (!Serial) delay(10);
  Serial.println("PIEZO_READY");
}

// Maximum gradient
float maxGradient() {
  int n = bufFull ? WINDOW : bufIdx;
  if (n < 2) return 0.0;
  float gMax = 0.0;
  for (int i = 1; i < n; i++) {
    int idxA = (bufIdx - n + i - 1 + WINDOW) % WINDOW;
    int idxB = (bufIdx - n + i     + WINDOW) % WINDOW;
    float diff = abs(buf[idxB] - buf[idxA]);
    if (diff > gMax) gMax = diff;
  }
  return gMax;
}

// Main loop
void loop() {
  int raw   = analogRead(PIN_PIEZO);
  float mv  = (float)raw / ADC_MAX * VCC_MV;

  // Update circular buffer
  buf[bufIdx] = mv;
  bufIdx = (bufIdx + 1) % WINDOW;
  if (bufIdx == 0) bufFull = true;

  unsigned long now = millis();

  if (mv > THRESHOLD_MV) {
    // Contact in progress — peak tracking
    if (mv > currentPeak) currentPeak = mv;
    inContact = true;
  } else if (inContact) {
    // Peak end → emit if cooldown respected
    if (now - lastPeakTime > COOLDOWN_MS) {
      float grad = maxGradient();
      // Send main peak
      Serial.print("IMPACT_MV:");
      Serial.println(currentPeak, 2);
      // Send gradient (optional, used by Python)
      Serial.print("GRADIENT:");
      Serial.println(grad, 2);
      lastPeakTime = now;
    }
    // Reset
    currentPeak = 0.0;
    inContact   = false;
    bufFull     = false;
    bufIdx      = 0;
  }

  delay(DELAY_MS);
}
