import node_class_mixins as mx
import requests, time, os, json

class ScenarioSeedance(mx.Node):
    CATEGORY = "Scenario"
    INPUTS   = ["prompt:str", "duration:int", "fps:int"]
    OUTPUTS  = ["video_url:str"]

    def run(self, prompt, duration=5, fps=24):
        auth = (os.getenv("SCENARIO_KEY"), os.getenv("SCENARIO_SECRET"))
        payload = {"prompt": prompt,
                   "duration": duration,
                   "fps": fps}
        r = requests.post(
            "https://api.cloud.scenario.com/v1/generate/custom/model_seedance-pro",
            auth=auth, json=payload)
        job_id = r.json()["job"]["jobId"]

        while True:
            status = requests.get(
               f"https://api.cloud.scenario.com/v1/jobs/{job_id}",
               auth=auth).json()["job"]
            if status["status"] == "success":
                asset_id = status["metadata"]["assetIds"][0]
                url = f"https://api.cloud.scenario.com/v1/assets/{asset_id}"
                return url
            if status["status"] == "failed":
                raise RuntimeError(status)
            time.sleep(2)
