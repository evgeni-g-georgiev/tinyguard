// ble.h — BLE communication between Node A (peripheral) and Node B (central).
//
// After training both nodes exchange validation statistics so Node B can
// calibrate the fused CUSUM. During monitoring Node A notifies its NLL
// score each clip; Node B reads it and fuses with its own.
//
// Val data packet (8 bytes):   mu_val + sigma_val  (fits in any BLE MTU)
// NLL score packet  (4 bytes): one float per clip
#pragma once
#include <ArduinoBLE.h>
#include <string.h>
#include "config.h"

#define NL_SERVICE_UUID  "19b10000-e8f2-537e-4f6c-d104768a1214"
#define NL_VALDATA_UUID  "19b10001-e8f2-537e-4f6c-d104768a1214"
#define NL_NLLSCORE_UUID "19b10002-e8f2-537e-4f6c-d104768a1214"

// Packet layout for the val-data characteristic.
struct ValDataPacket {
    float mu_val;
    float sigma_val;
};

// ── Node A: BLE Peripheral ────────────────────────────────────────────────────
#if NODE_ID == NODE_A

static BLEService        nl_service(NL_SERVICE_UUID);
static BLECharacteristic nl_valdata_char(NL_VALDATA_UUID,
                                         BLERead,
                                         sizeof(ValDataPacket));
static BLECharacteristic nl_nll_char(NL_NLLSCORE_UUID,
                                     BLERead | BLENotify,
                                     sizeof(float));

inline void ble_begin() {
    BLE.begin();
    BLE.setLocalName("TinyML-NodeA");
    BLE.setAdvertisedService(nl_service);
    nl_service.addCharacteristic(nl_valdata_char);
    nl_service.addCharacteristic(nl_nll_char);
    BLE.addService(nl_service);
    BLE.advertise();
    Serial.println("[BLE] Node A advertising.");
}

// Publish val stats once after calibrate() so Node B can read them in SYNC.
// det_val_nlls, det_mu_val, det_sigma_val are populated by detector.h.
inline void ble_publish_val_data() {
    ValDataPacket pkt;
    pkt.mu_val    = det_mu_val;
    pkt.sigma_val = det_sigma_val;
    nl_valdata_char.writeValue((const uint8_t*)&pkt, sizeof(pkt));
    Serial.println("[BLE] Val data published.");
}

// Notify Node B of this clip's NLL score.
inline void ble_send_nll(float nll) {
    nl_nll_char.writeValue((const uint8_t*)&nll, sizeof(float));
}

// Non-blocking BLE poll; returns true while a central is connected.
inline bool ble_poll_connected() {
    BLE.poll();
    return BLE.connected();
}

// ── Node B: BLE Central ───────────────────────────────────────────────────────
#else

static BLEDevice         nl_peripheral;
static BLECharacteristic nl_valdata_remote;
static BLECharacteristic nl_nll_remote;
static float             ble_last_nll  = 0.0f;
static bool              ble_nll_fresh = false;

inline void ble_begin() {
    BLE.begin();
    Serial.println("[BLE] Node B scanning for Node A...");
}

// Scan and connect to Node A — blocks until found or timeout (30 s).
// Returns true on success.
inline bool ble_connect() {
    BLE.scanForName("TinyML-NodeA");
    unsigned long t0 = millis();
    while (millis() - t0 < 30000UL) {
        BLE.poll();
        nl_peripheral = BLE.available();
        if (nl_peripheral) {
            BLE.stopScan();
            if (nl_peripheral.connect()) {
                nl_peripheral.discoverAttributes();
                nl_valdata_remote = nl_peripheral.characteristic(NL_VALDATA_UUID);
                nl_nll_remote     = nl_peripheral.characteristic(NL_NLLSCORE_UUID);
                nl_nll_remote.subscribe();
                Serial.println("[BLE] Connected to Node A.");
                return true;
            }
        }
    }
    Serial.println("[BLE] ERROR: Node A not found within 30 s.");
    return false;
}

// Read the 48-byte val-data packet from Node A. Call once in SYNC.
inline bool ble_read_val_data(ValDataPacket* out) {
    if (!nl_valdata_remote) return false;
    uint8_t buf[sizeof(ValDataPacket)];
    int n = nl_valdata_remote.readValue(buf, sizeof(buf));
    if (n != (int)sizeof(ValDataPacket)) return false;
    memcpy(out, buf, sizeof(ValDataPacket));
    return true;
}

// Poll BLE notifications — call each loop() tick in MONITOR.
// Sets ble_nll_fresh=true and updates ble_last_nll when Node A sends a score.
inline void ble_poll() {
    if (!nl_peripheral.connected()) return;
    BLE.poll();
    if (nl_nll_remote.valueUpdated()) {
        uint8_t buf[sizeof(float)];
        nl_nll_remote.readValue(buf, sizeof(buf));
        memcpy(&ble_last_nll, buf, sizeof(float));
        ble_nll_fresh = true;
    }
}

#endif // NODE_ID
