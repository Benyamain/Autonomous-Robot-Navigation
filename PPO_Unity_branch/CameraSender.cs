using UnityEngine;
using System.Collections;
using UnityEngine.Networking;
using System.IO;
using System;

public class CameraSender : MonoBehaviour
{
    public Camera cam;
    public string serverUrl = "http://localhost:5001/show";  //Flask server URL
    public float sendInterval = 0.5f;  // Interval between sending images

    public FourWheelAgent agent;  // Reference to the reinforcement learning Agent

    private void Start()
    {
        StartCoroutine(SendLoop());
    }

    IEnumerator SendLoop()
    {
        while (true)
        {
            yield return new WaitForSeconds(sendInterval);
            yield return StartCoroutine(SendImageToPython());
        }
    }

    IEnumerator SendImageToPython()
    {
        // Capture the camera image
        RenderTexture rt = new RenderTexture(416, 416, 24);
        cam.targetTexture = rt;
        Texture2D screenshot = new Texture2D(416, 416, TextureFormat.RGB24, false);
        cam.Render();
        RenderTexture.active = rt;
        screenshot.ReadPixels(new Rect(0, 0, 416, 416), 0, 0);
        screenshot.Apply();
        cam.targetTexture = null;
        RenderTexture.active = null;
        Destroy(rt);

        // Encode the image to Base64
        byte[] imageBytes = screenshot.EncodeToJPG();
        string imageBase64 = Convert.ToBase64String(imageBytes);
        string json = "{\"image\":\"" + imageBase64 + "\"}";

        // Send the image to the  Flask server
        using (UnityWebRequest www = UnityWebRequest.Put(serverUrl, json))
        {
            www.method = UnityWebRequest.kHttpVerbPOST;
            www.SetRequestHeader("Content-Type", "application/json");

            yield return www.SendWebRequest();

            if (www.result != UnityWebRequest.Result.Success)
            {
                Debug.LogError(" Failed to send YOLO inference request: " + www.error);
            }
            else
            {
                ProcessResponse(www.downloadHandler.text);
            }
        }
    }

    void ProcessResponse(string json)
    {
        try
        {
            YoloResponse response = JsonUtility.FromJson<YoloResponse>(json);
            if (response.target != null && response.target.label == "person")
            {
                int centerX = response.target.center_x;
                float offset = (centerX - 208f) / 208f;
                Debug.Log($" Detected person | Offset: {offset:F2}");
                agent?.UpdateTargetOffset(offset, true);
            }
            else
            {
                Debug.Log(" No person detected, passing 0 offset");
                agent?.UpdateTargetOffset(0f, false);
            }
        }
        catch (Exception e)
        {
            Debug.LogError($" Failed to parse JSON response: {e.Message}");
        }
    }

    // === JSON Response Structure ===
    [Serializable]
    public class YoloResponse
    {
        public string message;
        public float latency_ms;
        public float max_latency;
        public TargetData target;

        [Serializable]
        public class TargetData
        {
            public string label;
            public int center_x;
            public int center_y;
            public float confidence;
        }
    }
}
