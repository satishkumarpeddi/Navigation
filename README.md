Implementation Plan: Hand Gesture System Control
The goal is to create a Python application that uses the webcam to detect hand gestures and perform desktop window management tasks (minimize, maximize, close) as well as interact with the system.

Confirmed Requirements
We are building a pure gesture-based window control application. The user will not use their hand as a virtual mouse cursor to click things. Instead, distinct gestures will trigger specific window actions (minimize, maximize, close).

Proposed Architecture

1. Hand Tracking Layer
   Use MediaPipe Hands to detect 21 3D landmarks on the user's hand from the live webcam feed (cv2.VideoCapture).
2. Gesture Recognition Layer
   Map specific landmark positions to actions:
   Open Palm: Maximize active window.
   Fist: Minimize active window.
   Thumb Down (or Three Finger Pinch): Close application.
   Pinch (Index + Thumb): Open a specific application (e.g., File Explorer or a predefined app).
3. Action Execution Layer
   Use PyAutoGUI keyboard shortcuts (e.g., Win + Up for maximize, Win + Down for minimize, Alt + F4 for close) and direct OS system commands.
   Verification Plan
   Manual Verification
   Run the python script.
   Verify tracking latency and accuracy via OpenCV feed.
   Perform gestures in front of the camera and verify:
   Smooth cursor movement.
   Clicks and drag drop works.
   Window maximization, minimization, and closure work reliably.
