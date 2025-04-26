using UnityEngine;

public class Environment : MonoBehaviour
{
    public GameObject[] obstacles;
    public Transform obstacleArea;  // Parent object on the ground or in an obstructed area
    public TargetMover target;

    public float areaRadius = 5f;

    public void ResetEnvironment()
    {
        // reset initial position
        if (target != null)
        {
            target.SetRandomPosition();
        }

        // Random placement of obstacles
        foreach (GameObject obstacle in obstacles)
        {
            Vector3 randomPos = new Vector3(
                Random.Range(-areaRadius, areaRadius),
                0,
                Random.Range(-areaRadius, areaRadius)
            );
            obstacle.transform.localPosition = randomPos;

            float randomRot = Random.Range(0, 360);
            obstacle.transform.localRotation = Quaternion.Euler(0, randomRot, 0);
        }
    }
}
