using UnityEngine;

public class FourWheelDifferentialDrive : MonoBehaviour
{
    public Transform leftFrontWheel;
    public Transform leftBackWheel;
    public Transform rightFrontWheel;
    public Transform rightBackWheel;

    public float wheelSpeed = 1.5f;
    public float trackWidth = 0.4f;

    void Update()
    {
        float move = Input.GetAxis("Vertical");    // W/S
        float turn = Input.GetAxis("Horizontal");  // A/D

        float leftSpeed = move - turn;
        float rightSpeed = move + turn;

        float v = (leftSpeed + rightSpeed) * 0.5f * wheelSpeed;
        float w = (rightSpeed - leftSpeed) / trackWidth * wheelSpeed;

        float dt = Time.deltaTime;
        transform.position += transform.forward * v * dt;
        transform.Rotate(0, w * dt * Mathf.Rad2Deg, 0);

        float wheelRotDeg = v / 0.075f * Mathf.Rad2Deg * dt;

        // Make the wheel rotate around its own local x-axis (i.e. the red arrow)
        leftFrontWheel.transform.Rotate(transform.right, wheelRotDeg, Space.World);
        leftBackWheel.transform.Rotate(transform.right, wheelRotDeg, Space.World);
        rightFrontWheel.transform.Rotate(transform.right, wheelRotDeg, Space.World);
        rightBackWheel.transform.Rotate(transform.right, wheelRotDeg, Space.World);

    }
}
