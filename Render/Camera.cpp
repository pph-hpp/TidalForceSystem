#include "Camera.h"
#include <glm/gtc/matrix_transform.hpp>
#include <iostream>

namespace Render
{
    Camera::Camera(glm::vec3 position, glm::vec3 up, glm::vec3 front, float yaw, float pitch)
    : Front(front),
    MovementSpeed(SPEED),
    m_mouse_sensitivity(SENSITIVITY),
    Zoom(ZOOM)
    {
        this->Position = position;
        this->WorldUp = up;
        this->Yaw = yaw;
        this->Pitch = pitch;
        this->updateCameraVectors();
    }

    Camera::Camera(float posX, float posY, float posZ, float upX, float upY, float upZ, float yaw, float pitch) : Front(glm::vec3(0.0f, 0.0f, -1.0f)),
    MovementSpeed(SPEED),
    m_mouse_sensitivity(SENSITIVITY),
    Zoom(ZOOM)
    {
        this->Position = glm::vec3(posX, posY, posZ);
        this->WorldUp = glm::vec3(upX, upY, upZ);
        this->Yaw = yaw;
        this->Pitch = pitch;
        this->updateCameraVectors();
    }
    
    glm::mat4 Camera::GetViewMatrix()
    {
        return glm::lookAt(this->Position, this->Position + this->Front, this->Up);
        //return glm::lookAt(this->Position, { 0, 0, 0 }, this->Up);
        //float distance = 3.0f;                 // 相机到原点的距离
        //float tilt_deg = 25.0f;               // 倾角
        //float tilt_rad = glm::radians(tilt_deg);

        //// 方向向量：右上方（XY正方向）
        //glm::vec3 direction = glm::normalize(glm::vec3(1.0f, 1.0f, -tan(tilt_rad)));

        //// 相机位置 = 距离 × 方向
        //glm::vec3 eye = distance * direction;

        //// 相机看向原点
        //glm::vec3 center = glm::vec3(0.0f, 0.0f, 0.0f);

        //// 保持 up 向量朝向 Z 轴正方向（避免上下翻转）
        //glm::vec3 up = glm::vec3(0.0f, 0.0f, 1.0f);

        //return glm::lookAt(eye, center, up);
    }

    void Camera::ProcessKeyboard(Camera_Movement direction, float deltaTime)
    {
        float velocity = this->MovementSpeed * deltaTime;
        if (direction == FORWARD)
            this->Position += this->Front * velocity;
        if (direction == BACKWARD)
            this->Position -= this->Front * velocity;
        if (direction == LEFT)
            this->Position -= this->Right * velocity;
        if (direction == RIGHT)
            this->Position += this->Right * velocity;
        if (direction == UP)
            this->Position += this->Up * velocity;
        if (direction == DOWN)
            this->Position -= this->Up * velocity;
    }

    void Camera::ProcessMouseMovement(float xoffset, float yoffset, bool constrainPitch)
    {
        xoffset = xoffset * m_mouse_sensitivity;
        yoffset = yoffset * m_mouse_sensitivity;
        Yaw += xoffset;
        Pitch += yoffset;
        if (constrainPitch){   
            if (Pitch > 89.0f)
                Pitch = 89.0f;
            if (Pitch < -89.0f)
                Pitch = -89.0f;
        }
        updateCameraVectors();
    }
    
    void Camera::ProcessMouseScroll(float yoffset)
    {
        this->Zoom -= (float)yoffset;
        if (this->Zoom < 1.0f)
            this->Zoom = 1.0f;
        if (this->Zoom > 45.0f)
            this->Zoom = 45.0f;
    }

    void Camera::updateCameraVectors()
    {
        glm::vec3 front;
        front.x = cos(glm::radians(this->Yaw)) * cos(glm::radians(this->Pitch));
        front.y = sin(glm::radians(this->Pitch));
        front.z = sin(glm::radians(this->Yaw)) * cos(glm::radians(this->Pitch));
        this->Front = glm::normalize(front);
        this->Right = glm::normalize(glm::cross(this->Front, this->WorldUp));
        this->Up = glm::normalize(glm::cross(this->Right, this->Front));
    }

    void Camera::setMoveSpeed(float speed)
    {
        this->MovementSpeed = speed;
    }

    void Camera::setMouseSensitivity(float sensitivity)
    {
        this->m_mouse_sensitivity = sensitivity;
    }

    void Camera::setZoom(float zoom)
    {
        this->Zoom = zoom;
    }
    
    
}
