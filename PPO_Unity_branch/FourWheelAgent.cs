using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using System.Collections;
using TMPro;

public class FourWheelAgent : Agent
{
    // Variables for target detection timing
    private float timeSinceTargetSeen = 0f;
    private const float maxTimeNearTarget = 3f;

    // Variables for detecting being stuck
    private Vector3 lastPosition;
    private float timeStuck = 0f;
    private float stuckThreshold = 3f;      
    private float minMoveDistance = 0.05f;  

    private float episodeTimer = 0f;
    public float maxEpisodeTime = 99999f;

    [Header("UI Elements")]
    public TextMeshProUGUI infoText;

    [Header("Debug Settings")]
    public bool pauseAfterEpisode = true;

    private int successCount = 0;
    private int wallHitCount = 0;
    private int obstacleHitCount = 0;

    private float cumulativeReward = 0f;
    private float currentDistance = 0f;

    private bool justPaused = false;
    private bool hasMovedTowardTarget = false;

    [Header("Wheel Transforms")]
    public Transform leftFrontWheel;
    public Transform leftBackWheel;
    public Transform rightFrontWheel;
    public Transform rightBackWheel;

    [Header("Target and Environment")]
    public Transform target;
    public Environment environment;

    [Header("Control Parameters")]
    public float wheelForce = 50f;
    public float wheelRadius = 0.075f;
    public float trackWidth = 0.4f;

    private Rigidbody rb;
    private Vector3 prevToTarget;

    void Start()
    {
        rb = GetComponent<Rigidbody>();
    }

    private float targetOffsetX = 0f; 
    public bool lastTargetSeen = false;

    public void UpdateTargetOffset(float offset, bool seen)
    {
        targetOffsetX = offset;
        lastTargetSeen = seen;
    }

    void OnDrawGizmos()
    {
        if (target == null) return;
        Gizmos.color = Color.green;
        Gizmos.DrawLine(transform.position, target.position);
    }

    private bool collisionEnabled = false;

    private IEnumerator EnableCollisionAfterDelay(float delay)
    {
        collisionEnabled = false;
        yield return new WaitForSeconds(delay);
        collisionEnabled = true;
    }

    public override void OnEpisodeBegin()
    {
        lastPosition = transform.position;
        timeStuck = 0f;

        Debug.Log("Episode Begin");
        episodeTimer = 0f;
        hasMovedTowardTarget = false;
        Time.timeScale = 1f;
        justPaused = false;

        environment.ResetEnvironment();
        transform.position = new Vector3(0f, 0.5f, -0.4f);
        transform.rotation = Quaternion.identity;

        rb.velocity = Vector3.zero;
        rb.angularVelocity = Vector3.zero;

        prevToTarget = target.position - transform.position;
        StartCoroutine(EnableCollisionAfterDelay(0.5f));
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        float forwardSpeed = Mathf.Clamp(actions.ContinuousActions[0], 0f, 1f);
        float turnRate = Mathf.Clamp(actions.ContinuousActions[1], -1f, 1f);

        float leftPower = Mathf.Clamp(forwardSpeed - turnRate, 0f, 1f);
        float rightPower = Mathf.Clamp(forwardSpeed + turnRate, 0f, 1f);

        Vector3 leftForce = transform.forward * leftPower * wheelForce;
        Vector3 rightForce = transform.forward * rightPower * wheelForce;

        rb.AddForceAtPosition(leftForce, leftFrontWheel.position, ForceMode.Force);
        rb.AddForceAtPosition(rightForce, rightFrontWheel.position, ForceMode.Force);

        // Wheel rotation visualization
        float avgSpeed = (leftPower + rightPower) * 0.5f;
        float deg = avgSpeed / wheelRadius * Mathf.Rad2Deg * Time.deltaTime;
        leftFrontWheel.Rotate(transform.right, deg, Space.World);
        leftBackWheel.Rotate(transform.right, deg, Space.World);
        rightFrontWheel.Rotate(transform.right, deg, Space.World);
        rightBackWheel.Rotate(transform.right, deg, Space.World);

        // Reward function
        Vector3 toTarget = target.position - transform.position;
        float distNow = toTarget.magnitude;
        float distPrev = prevToTarget.magnitude;
        float distanceDelta = Mathf.Clamp(distPrev - distNow, -1f, 1f);
        float angle = Vector3.Angle(transform.forward, toTarget.normalized);
        float dot = Vector3.Dot(transform.forward, toTarget.normalized);

        float rewardDist = (distanceDelta > 0) ? distanceDelta * 0.5f : 0f;
        float rewardAlign = (1f - Mathf.Abs(targetOffsetX)) * 0.1f;
        float rewardAngle = -Mathf.Clamp01(angle / 90f) * 0.02f;
        float rewardTime = -0.001f;
        float rewardReverse = (dot < 0f) ? -0.05f : 0f;

        AddReward(rewardDist + rewardAlign + rewardAngle + rewardTime + rewardReverse);

        prevToTarget = toTarget;
        episodeTimer += Time.deltaTime;

        // Successful approach
        if (distNow < 1.0f && lastTargetSeen && Mathf.Abs(targetOffsetX) < 0.3f)
        {
            successCount++;
            AddReward(+2f);
            Debug.Log("Successfully reached the target, ending episode.");
            EndEpisode();
            return;
        }

        // Timeout
        if (episodeTimer >= maxEpisodeTime)
        {
            AddReward(-1f);
            Debug.Log("Episode timeout, ending episode.");
            EndEpisode();
            return;
        }

        // Fail to approach after detecting the target
        if (lastTargetSeen)
        {
            timeSinceTargetSeen += Time.deltaTime;

            if (timeSinceTargetSeen > maxTimeNearTarget && distNow > 1.5f)
            {
                AddReward(-1f);
                Debug.Log("Detected target but failed to approach, ending episode.");
                EndEpisode();
                return;
            }
        }
        else
        {
            timeSinceTargetSeen = 0f;
        }

        // Anti-stuck mechanism
        float movedDistance = Vector3.Distance(transform.position, lastPosition);

        if (movedDistance < minMoveDistance)
        {
            timeStuck += Time.deltaTime;

            if (timeStuck >= stuckThreshold)
            {
                AddReward(-1f);
                Debug.Log($"Agent stuck for more than {stuckThreshold:F1}s, ending episode.");
                EndEpisode();
                return;
            }
        }
        else
        {
            timeStuck = 0f;
            lastPosition = transform.position;
        }

        // Update UI (optional)
        cumulativeReward = GetCumulativeReward();
        currentDistance = distNow;
        if (infoText != null)
        {
            infoText.text = $"Distance: {currentDistance:F2}m\n" +
                            $"Reward: {cumulativeReward:F3}\n" +
                            $"Success: {successCount}\n" +
                            $"Hit wall: {wallHitCount}\n" +
                            $"Hit Obstacle: {obstacleHitCount}";
        }
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        Debug.Log("Manual control mode activated.");
        var ca = actionsOut.ContinuousActions;
        float move = Input.GetAxis("Vertical");
        float turn = Input.GetAxis("Horizontal");
        ca[0] = move - turn;
        ca[1] = move + turn;
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        // YOLO offset and visibility
        sensor.AddObservation(targetOffsetX);
        sensor.AddObservation(lastTargetSeen ? 1f : 0f);

        // Lidar distances at several angles
        float[] angles = { -60f, -30f, 0f, 30f, 60f };
        foreach (float angle in angles)
        {
            Vector3 dir = Quaternion.Euler(0, angle, 0) * transform.forward;
            if (Physics.Raycast(transform.position, dir, out RaycastHit hit, 5f))
                sensor.AddObservation(hit.distance / 5f);
            else
                sensor.AddObservation(1f);
        }

        // Agent orientation and speed
        sensor.AddObservation(transform.forward);
        sensor.AddObservation(rb.velocity.magnitude / 5f);
    }

    private void OnCollisionEnter(Collision col)
    {
        if (!collisionEnabled || justPaused) return;

        string tag = col.collider.tag;

        switch (tag)
        {
            case "Wall":
                wallHitCount++;
                AddReward(-1f);
                Debug.Log($"Hit wall. Current reward: {GetCumulativeReward():F3}");
                break;

            case "Obstacle":
                obstacleHitCount++;
                AddReward(-1f);
                Debug.Log($"Hit obstacle. Current reward: {GetCumulativeReward():F3}");
                break;

            case "Target":
                successCount++;
                AddReward(+2f);
                Debug.Log($"Hit target. Current reward: {GetCumulativeReward():F3}");
                break;

            default:
                return;
        }

        if (pauseAfterEpisode)
        {
            Time.timeScale = 0f;
            justPaused = true;
        }

        EndEpisode();
    }
}
