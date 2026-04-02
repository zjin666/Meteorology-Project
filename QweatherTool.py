import requests
from typing import Dict, Optional, Union, Tuple

class WeatherAPITool:
    """
    封装和风天气API的工具类，供大语言模型通过函数调用（function calling）方式使用。
    所有方法均为无状态，依赖外部传入的 API 密钥与基础 URL。
    返回格式统一为：{'status': 'success', 'data': {...}} 或 {'status': 'error', 'message': '...'}
    """

    def __init__(self, api_key: str):
        """
        初始化工具类
        
        :param api_key: 和风天气 API 密钥
        :param base_url: API 基础地址，默认为和风天气 v3 接口
        """
        self.api_key = api_key
        self.url_api_weather = 'https://n23wt39cft.re.qweatherapi.com/v7/weather/'
        self.url_api_geo = 'https://n23wt39cft.re.qweatherapi.com/geo/v2/city/'
        self.url_api_rain = 'https://n23wt39cft.re.qweatherapi.com/v7/minutely/5m'
        self.url_api_air = 'https://n23wt39cft.re.qweatherapi.com/airquality/v1/current/'

    def get_city_info(self, city_keyword: str) -> Dict[str, Union[str, Dict]]:
        """
        根据城市关键词查询地理信息（ID、经纬度、行政区划）

        :param city_keyword: 城市名称，如 "杭州"
        :return: 包含城市信息的字典或错误信息
        """
        try:
            url = self.url_api_geo + 'lookup?location=' + city_keyword
            # 请求头
            headers = {
                "X-QW-Api-Key": self.api_key,
                "Accept-Encoding": "gzip, deflate, br"  # 对应curl的--compressed参数
            }
            response = requests.get(
                url=url,
                headers=headers)
            response.raise_for_status()
            data = response.json()

            if not data.get("location"):
                return {"status": "error", "message": "未找到该城市信息"}

            loc = data["location"][0]
            return {
                "status": "success",
                "data": {
                    "city_id": loc["id"],
                    "district_name": loc["name"],
                    "city_name": loc["adm2"],
                    "province_name": loc["adm1"],
                    "country_name": loc["country"],
                    "lat": loc["lat"],
                    "lon": loc["lon"]
                }
            }
        except Exception as e:
            return {"status": "error", "message": f"请求失败: {str(e)}"}

    def get_weather_now(self, location_id: str) -> Dict[str, Union[str, Dict]]:
        """
        获取指定城市的实时天气

        :param location_id: 城市ID，由 get_city_info 返回
        :return: 实时天气数据
        """
        try:
            url = self.url_api_weather + 'now' + '?location=' + location_id
            # 请求头
            headers = {
                "X-QW-Api-Key": self.api_key,
                "Accept-Encoding": "gzip, deflate, br"  # 对应curl的--compressed参数
            }
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()

            if not data.get("now"):
                return {"status": "error", "message": "获取实时天气失败"}

            now = data["now"]
            return {
                "status": "success",
                "data": {
                    "temperature": now["temp"],
                    "humidity": now["humidity"],
                    "wind_speed": now["windSpeed"],
                    "weather_text": now["text"],
                    "last_update": data["updateTime"]
                }
            }
        except Exception as e:
            return {"status": "error", "message": f"获取实时天气失败: {str(e)}"}

    def get_weather_daily(self, location_id: str, days: str = "3d") -> Dict[str, Union[str, Dict]]:
        """
        获取指定城市的多日天气预报（支持 3d / 7d / 10d / 15d）

        :param location_id: 城市ID
        :param days: 预报天数，默认 "3d"
        :return: 多日天气数据
        """
        try:
            url = self.url_api_weather + days + '?location=' + location_id
            # 请求头
            headers = {
                "X-QW-Api-Key": self.api_key,
                "Accept-Encoding": "gzip, deflate, br"  # 对应curl的--compressed参数
            }
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()

            if not data.get("daily"):
                return {"status": "error", "message": "获取预报失败"}

            daily = data["daily"]
            return {
                "status": "success",
                "data": daily
            }
        except Exception as e:
            return {"status": "error", "message": f"获取预报失败: {str(e)}"}

    def get_weather_hourly(self, location_id: str, hours: str = "24h") -> Dict[str, Union[str, Dict]]:
        """
        获取指定城市的逐小时天气预报（支持 24h / 72h / 168h）

        :param location_id: 城市ID
        :param hours: 预报时长，默认 "24h"
        :return: 逐小时天气数据
        """
        try:
            url = self.url_api_weather + hours + '?location=' + location_id
            # 请求头
            headers = {
                "X-QW-Api-Key": self.api_key,
                "Accept-Encoding": "gzip, deflate, br"  # 对应curl的--compressed参数
            }
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()

            if not data.get("hourly"):
                return {"status": "error", "message": "获取小时预报失败"}

            hourly = data["hourly"]
            return {
                "status": "success",
                "data": hourly
            }
        except Exception as e:
            return {"status": "error", "message": f"获取小时预报失败: {str(e)}"}

    def get_rain_forecast(self, latitude: str, longitude: str) -> Dict[str, Union[str, Dict]]:
        """
        获取指定经纬度的分钟级降水预报

        :param latitude: 纬度，字符串格式，如 "30.2741"
        :param longitude: 经度，字符串格式，如 "120.1541"
        :return: 降水预报数据
        """
        try:
            url = self.url_api_rain  +'?location=' + latitude + ',' + longitude
            # 请求头
            headers = {
                "X-QW-Api-Key": self.api_key,
                "Accept-Encoding": "gzip, deflate, br"  # 对应curl的--compressed参数
            }
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()

            if not data.get("summary"):
                return {"status": "error", "message": "获取降水预报失败"}

            rain_data = data.get("minutely", {})
            return {
                "status": "success",
                "summary": data["summary"],
                "data": rain_data
            }
        except Exception as e:
            return {"status": "error", "message": f"获取降水预报失败: {str(e)}"}

    #def get_air_quality_now(self, latitude: str, longitude: str) -> Dict[str, Union[str, Dict]]:
        """
        获取指定城市的当前空气质量（AQI、PM2.5、PM10等）

        :param location_id: 城市ID
        :return: 空气质量数据
        """
        try:
            url = self.url_api_air + latitude + '/' + longitude
            params = {"key": self.api_key, "location": location_id}
            response = requests.get(url, params=params, timeout=5)
            response.raise_for_status()
            data = response.json()

            if data.get("status") != "ok":
                return {"status": "error", "message": "获取空气质量失败"}

            air = data["results"][0]["air"]
            return {
                "status": "success",
                "data": {
                    "aqi": air["aqi"],
                    "pm25": air["pm25"],
                    "pm10": air["pm10"],
                    "so2": air["so2"],
                    "no2": air["no2"],
                    "co": air["co"],
                    "o3": air["o3"],
                    "quality": air["quality"],
                    "time": data["results"][0]["last_update"]
                }
            }
        except Exception as e:
            return {"status": "error", "message": f"获取空气质量失败: {str(e)}"}
