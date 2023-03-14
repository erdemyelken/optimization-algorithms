# import pandas as pd
# import matplotlib.pyplot as plt
# import json

# df = pd.read_csv('prtg_factory_sensors.csv', header=None, names=['sensor_id', 'name', 'description', 'data', 'start_time', 'end_time'])

# df['data'] = df['data'].apply(lambda x: json.loads(x))
# df = pd.json_normalize(df['data'])

# plt.plot(df['traffic_in_speed'], label='Traffic In Speed')
# plt.plot(df['traffic_out_speed'], label='Traffic Out Speed')
# plt.legend()
# plt.show()


# import pandas as pd
# import json
# import matplotlib.pyplot as plt

# # CSV dosyasını yükle
# df = pd.read_csv('prtg_factory_sensors.csv')

# # metrics_json sütunundaki JSON verilerini ayrıştır
# df['metrics_json'] = df['metrics_json'].apply(json.loads)

# # "traffic_out_volume" ve "traffic_in_volume" sütunlarını seç
# df = df[['traffic_out_volume', 'traffic_in_volume']]

# # "traffic_out_volume" ve "traffic_in_volume" sütunlarındaki verileri grafikleştir
# df.plot(kind='bar')
# plt.show()

# prtg_network.csv

# import pandas as pd

# df = pd.read_csv('prtg_network.csv')
# print(df.head())

# import matplotlib.pyplot as plt

# # datetime formatını belirle
# df['start_date'] = pd.to_datetime(df['start_date'])
# # saat başına toplam trafik miktarını hesapla
# df['hourly_traffic'] = df['traffic_total_speed'] / (1024 * 1024)

# # saat başına toplam trafik miktarını gösteren çizgi grafiği oluştur
# plt.plot(df['start_date'], df['hourly_traffic'])
# plt.xlabel('Tarih')
# plt.ylabel('Saatlik Trafik (MB)')
# plt.title('Saatlik Trafik Değişimi')
# plt.show()

import pandas as pd
import matplotlib.pyplot as plt

# Veri setini okuyun
df = pd.read_csv('prtg_factory_sensors.csv')

# metrics_json sütununu JSON veri tipine dönüştürün
df['metrics_json'] = df['metrics_json'].apply(lambda x: eval(x))

# Grafik için veri hazırlığı
start_dates = pd.to_datetime(df['start_date'])
end_dates = pd.to_datetime(df['end_date'])
traffic_out_volumes = [x['traffic_out_volume'] for x in df['metrics_json']]
traffic_in_volumes = [x['traffic_in_volume'] for x in df['metrics_json']]

# Grafik çizme
fig, ax = plt.subplots()
ax.plot(start_dates, traffic_out_volumes, label='Giden Trafik')
ax.plot(start_dates, traffic_in_volumes, label='Gelen Trafik')
ax.set_xlabel('Tarih')
ax.set_ylabel('Trafik Miktarı')
ax.set_title('PRTG Trafik Ölçümleri')
ax.legend()
plt.show()