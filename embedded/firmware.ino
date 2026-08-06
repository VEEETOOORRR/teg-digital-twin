#include <WiFi.h>
#include <PubSubClient.h>
#include <Wire.h>
#include <Adafruit_INA219.h>
#include <Adafruit_ADS1X15.h>
#include <ArduinoJson.h>
#include <cmath>

// --- CONFIGURAÇÕES DE REDE E MQTT ---
const char* ssid = "NOME_DA_SUA_REDE_WIFI";
const char* password = "SENHA_DO_SEU_WIFI";
const char* mqtt_server = "192.168.X.X"; // Substitua pelo IP Local da sua máquina rodando o Docker
const int mqtt_port = 1883;
const char* mqtt_topic = "teg/bancada/telemetria";

// --- PARÂMETROS DOS SENSORES NTC 100K 3950 ---
const float SERIES_RESISTOR = 100000.0; // Resistor de 100k em série
const float NOMINAL_RESISTANCE = 100000.0; // Resistência nominal do NTC a 25C
const float NOMINAL_TEMPERATURE = 25.0; // Temp nominal em Celsius
const float B_COEFFICIENT = 3950.0; // Coeficiente Beta do NTC

// --- INSTÂNCIAS DOS COMPONENTES ---
WiFiClient espClient;
PubSubClient client(espClient);
Adafruit_INA219 ina219;
Adafruit_ADS1115 ads;

unsigned long lastMsg = 0;

// Função para calcular a temperatura do NTC via Steinhart-Hart
float calcularTemperaturaNTC(int16_t adc_value) {
  // O ADS1115 em modo single-ended padrão vai até 4.096V para 32767 de leitura
  // Mas como estamos alimentando com 3.3V, a leitura máxima teórica será menor.
  if (adc_value <= 0) return 0.0;
  
  // Calcular a resistência do NTC
  float v_out = adc_value * (4.096 / 32768.0);
  float r_ntc = SERIES_RESISTOR * (3.3 / v_out - 1.0);
  
  // Equação simplificada de Steinhart-Hart (Equação Beta)
  float steinhart;
  steinhart = r_ntc / NOMINAL_RESISTANCE;     // (R/Ro)
  steinhart = log(steinhart);                  // ln(R/Ro)
  steinhart /= B_COEFFICIENT;                  // 1/B * ln(R/Ro)
  steinhart += 1.0 / (NOMINAL_TEMPERATURE + 273.15); // + (1/To)
  steinhart = 1.0 / steinhart;                 // Inverter para obter Kelvin
  steinhart -= 273.15;                         // Converter para Celsius
  
  return steinhart;
}

void setup_wifi() {
  delay(10);
  Serial.println();
  Serial.print("Conectando a rede: ");
  Serial.println(ssid);

  WiFi.begin(ssid, password);

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println("\nWiFi conectado!");
  Serial.print("Endereço IP: ");
  Serial.println(WiFi.localIP());
}

void reconnect() {
  while (!client.connected()) {
    Serial.print("Tentando conexão MQTT...");
    // Tenta conectar com um ID único baseado no MAC address
    String clientId = "ESP32-TEG-Client-";
    clientId += String(random(0xffff), HEX);
    
    if (client.connect(clientId.c_str())) {
      Serial.println("conectado com sucesso!");
    } else {
      Serial.print("falhou, rc=");
      Serial.print(client.state());
      Serial.println(" Tentando novamente em 5 segundos.");
      delay(5000);
    }
  }
}

void setup() {
  Serial.begin(115200);
  Wire.begin(21, 22); // Inicializa I2C nos pinos padrão do ESP32 (SDA=21, SCL=22)

  setup_wifi();
  client.setServer(mqtt_server, mqtt_port);

  // Inicializa INA219
  if (!ina219.begin()) {
    Serial.println("Falha ao encontrar o chip INA219");
  }
  
  // Inicializa ADS1115
  ads.setGain(GAIN_ONE); // Ganho 1x para ler até 4.096V
  if (!ads.begin()) {
    Serial.println("Falha ao encontrar o chip ADS1115");
  }
}

void loop() {
  if (!client.connected()) {
    reconnect();
  }
  client.loop();

  unsigned long now = millis();
  // Executa o envio a cada 1000 milissegundos (1 segundo)
  if (now - lastMsg > 1000) {
    lastMsg = now;

    // 1. Leituras elétricas do TEG (INA219)
    float shuntvoltage = ina219.getShuntVoltage_mV();
    float busvoltage = ina219.getBusVoltage_V();
    float current_mA = ina219.getCurrent_mA();
    float power_mW = ina219.getPower_mW();
    float loadvoltage = busvoltage + (shuntvoltage / 1000);

    // 2. Leituras térmicas dos NTCs via ADS1115
    int16_t adc0 = ads.readADC_SingleEnded(0);
    int16_t adc1 = ads.readADC_SingleEnded(1);
    int16_t adc2 = ads.readADC_SingleEnded(2);
    int16_t adc3 = ads.readADC_SingleEnded(3);

    // Converte leituras brutas em graus Celsius
    // T1 e T2 posicionados na face superior (Quente) | T3 e T4 na face inferior (Fria)
    float t_quente_1 = calcularTemperaturaNTC(adc0);
    float t_quente_2 = calcularTemperaturaNTC(adc1);
    float t_frio_1   = calcularTemperaturaNTC(adc2);
    float t_frio_2   = calcularTemperaturaNTC(adc3);

    // Médias para robustez do cálculo do Gêmeo Digital
    float t_quente_media = (t_quente_1 + t_quente_2) / 2.0;
    float t_frio_media = (t_frio_1 + t_frio_2) / 2.0;
    float delta_t = t_quente_media - t_frio_media;

    // 3. Montagem do payload JSON usando ArduinoJson v6/v7
    StaticJsonDocument<300> doc;
    doc["device_id"] = "Bancada_TEG_01";
    
    // Sub-objeto elétrico
    JsonObject eletrico = doc.createNestedObject("eletrico");
    eletrico["tensao_V"] = loadvoltage;
    eletrico["corrente_mA"] = current_mA;
    eletrico["potencia_mW"] = power_mW;

    // Sub-objeto térmico
    JsonObject termico = doc.createNestedObject("termico");
    termico["t_quente"] = t_quente_media;
    path_termico["t_frio"] = t_frio_media;
    termico["delta_t"] = delta_t;

    // Converte o objeto JSON para String
    char buffer[300];
    serializeJson(doc, buffer);

    // 4. Publicação via MQTT Broker local
    Serial.print("Publicando dados: ");
    Serial.println(buffer);
    client.publish(mqtt_topic, buffer);
  }
}