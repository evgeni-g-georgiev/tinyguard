// ble.h — BLE communication between Node A (peripheral) and Node B (central).
//
// All four GATT characteristics are hosted on Node A's peripheral server.
// Node B (central) reads/writes to Node A's characteristics in both directions.
//
// Characteristic layout (all on Node A's GATT server):
//   NL_VALDATA_A  : BLERead         — Node A's val stats + val NLLs (B reads in SYNC)
//   NL_NLLSCORE_A : BLERead|Notify  — Node A's per-clip NLL (B subscribes in MONITOR)
//   NL_VALDATA_B  : BLEWrite        — Node B writes its val stats + val NLLs (A reads in SYNC)
//   NL_NLLSCORE_B : BLEWrite        — Node B writes its per-clip NLL (A reads in MONITOR)
//
// ValDataPacket: mu_val + sigma_val + val_nlls[N_VAL_CLIPS]
//   = 8 + 4*N_VAL_CLIPS bytes = 48 bytes at N_VAL_CLIPS=10.
//   ArduinoBLE handles multi-packet reads transparently.
// NLL score packet: 4 bytes (one float).
#pragma once
#include <ArduinoBLE.h>
#include <string.h>
#include "config.h"

#define NL_SERVICE_UUID    "19b10000-e8f2-537e-4f6c-d104768a1214"
#define NL_VALDATA_A_UUID  "19b10001-e8f2-537e-4f6c-d104768a1214"
#define NL_NLLSCORE_A_UUID "19b10002-e8f2-537e-4f6c-d104768a1214"
#define NL_VALDATA_B_UUID  "19b10003-e8f2-537e-4f6c-d104768a1214"
#define NL_NLLSCORE_B_UUID "19b10004-e8f2-537e-4f6c-d104768a1214"

struct ValDataPacket {
    float mu_val;
    float sigma_val;
    float val_nlls[N_VAL_CLIPS];
    float chosen_r;   // r selected by this node's r-search; used for diversity constraint
};

// ── Node A: BLE Peripheral ────────────────────────────────────────────────────
#if NODE_ID == NODE_A

static BLEService        nl_service(NL_SERVICE_UUID);
static BLECharacteristic nl_valdata_a_char(NL_VALDATA_A_UUID, BLERead,              sizeof(ValDataPacket));
static BLECharacteristic nl_nlla_char     (NL_NLLSCORE_A_UUID, BLERead | BLENotify, sizeof(float));
static BLECharacteristic nl_valdata_b_char(NL_VALDATA_B_UUID, BLEWrite,             sizeof(ValDataPacket));
static BLECharacteristic nl_nllb_char     (NL_NLLSCORE_B_UUID, BLEWrite,            sizeof(float));

static float ble_last_nll_b  = 0.0f;
static bool  ble_nll_b_fresh = false;

inline void ble_begin() {
    BLE.begin();
    BLE.setLocalName("TinyML-NodeA");
    BLE.setAdvertisedService(nl_service);
    nl_service.addCharacteristic(nl_valdata_a_char);
    nl_service.addCharacteristic(nl_nlla_char);
    nl_service.addCharacteristic(nl_valdata_b_char);
    nl_service.addCharacteristic(nl_nllb_char);
    BLE.addService(nl_service);
    BLE.advertise();
    Serial.println("[BLE] Node A advertising.");
}

// Publish our val stats so Node B can read them during SYNC.
inline void ble_publish_val_data(float mu_val, float sigma_val, const float* val_nlls, float r) {
    ValDataPacket pkt;
    pkt.mu_val    = mu_val;
    pkt.sigma_val = sigma_val;
    memcpy(pkt.val_nlls, val_nlls, N_VAL_CLIPS * sizeof(float));
    pkt.chosen_r  = r;
    nl_valdata_a_char.writeValue((const uint8_t*)&pkt, sizeof(pkt));
    Serial.println("[BLE] Val data published.");
}

// Notify Node B of this clip's NLL score.
inline void ble_send_nll(float nll) {
    nl_nlla_char.writeValue((const uint8_t*)&nll, sizeof(float));
}

// Poll BLE; if Node B wrote its val stats, copy them into *out and return true.
// Edge-triggered: returns true at most once per write event.
inline bool ble_read_val_data_b(ValDataPacket* out) {
    BLE.poll();
    if (!nl_valdata_b_char.written()) return false;
    uint8_t buf[sizeof(ValDataPacket)];
    nl_valdata_b_char.readValue(buf, sizeof(buf));
    memcpy(out, buf, sizeof(ValDataPacket));
    return true;
}

// Poll BLE; if Node B wrote its NLL, update ble_last_nll_b and set ble_nll_b_fresh.
inline void ble_poll_nll_b() {
    BLE.poll();
    if (nl_nllb_char.written()) {
        nl_nllb_char.readValue(&ble_last_nll_b, sizeof(float));
        ble_nll_b_fresh = true;
    }
}

inline bool ble_is_connected() { BLE.poll(); return BLE.connected(); }

// ── Node B: BLE Central ───────────────────────────────────────────────────────
#else

static BLEDevice         nl_peripheral;
static BLECharacteristic nl_valdata_a_remote;
static BLECharacteristic nl_nlla_remote;
static BLECharacteristic nl_valdata_b_remote;
static BLECharacteristic nl_nllb_remote;
static float             ble_last_nll  = 0.0f;
static bool              ble_nll_fresh = false;

inline void ble_begin() {
    BLE.begin();
    Serial.println("[BLE] Node B ready.");
}

// Scan for Node A; connect and discover all four characteristics.
// Returns true on success; false on 30-second timeout.
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
                nl_valdata_a_remote = nl_peripheral.characteristic(NL_VALDATA_A_UUID);
                nl_nlla_remote      = nl_peripheral.characteristic(NL_NLLSCORE_A_UUID);
                nl_valdata_b_remote = nl_peripheral.characteristic(NL_VALDATA_B_UUID);
                nl_nllb_remote      = nl_peripheral.characteristic(NL_NLLSCORE_B_UUID);
                nl_nlla_remote.subscribe();
                Serial.println("[BLE] Connected to Node A.");
                return true;
            }
        }
    }
    Serial.println("[BLE] Node A not found within 30 s.");
    return false;
}

// Read Node A's val-data packet. Call once after ble_connect() in SYNC.
inline bool ble_read_val_data(ValDataPacket* out) {
    if (!nl_valdata_a_remote) return false;
    uint8_t buf[sizeof(ValDataPacket)];
    int n = nl_valdata_a_remote.readValue(buf, sizeof(buf));
    if (n != (int)sizeof(ValDataPacket)) return false;
    memcpy(out, buf, sizeof(ValDataPacket));
    return true;
}

// Write our val-data packet to Node A's NL_VALDATA_B characteristic.
inline bool ble_write_val_data(const ValDataPacket* pkt) {
    if (!nl_valdata_b_remote) return false;
    return nl_valdata_b_remote.writeValue((const uint8_t*)pkt, sizeof(ValDataPacket));
}

// Poll BLE notifications from Node A. Sets ble_nll_fresh=true when Node A sends a score.
inline void ble_poll() {
    if (!nl_peripheral.connected()) return;
    BLE.poll();
    if (nl_nlla_remote.valueUpdated()) {
        uint8_t buf[sizeof(float)];
        nl_nlla_remote.readValue(buf, sizeof(buf));
        memcpy(&ble_last_nll, buf, sizeof(float));
        ble_nll_fresh = true;
    }
}

// Write our per-clip NLL to Node A's NL_NLLSCORE_B characteristic.
inline void ble_send_nll_b(float nll) {
    if (!nl_peripheral.connected()) return;
    nl_nllb_remote.writeValue((const uint8_t*)&nll, sizeof(float));
}

inline bool ble_is_connected() { return nl_peripheral.connected(); }

#endif // NODE_ID
