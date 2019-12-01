# Assignment-23
dlib, object tracking with 68 face landmarks with video landmark stabilization


Video link:

https://www.youtube.com/watch?v=K-RFjRfKaWE&feature=youtu.be

<a href="http://www.youtube.com/watch?feature=player_embedded&v=https://www.youtube.com/watch?v=K-RFjRfKaWE&feature=youtu.be
" target="_blank"><img src="https://www.youtube.com/watch?v=K-RFjRfKaWE&feature=youtu.be%2F3.jpg" 
alt="Video of face detection and tracking." width="240" height="180" border="10" /></a>


[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/https://www.youtube.com/watch?v=K-RFjRfKaWE&feature=youtu.be/0.jpg)](https://www.youtube.com/watch?v=https://www.youtube.com/watch?v=K-RFjRfKaWE&feature=youtu.be)


PIPE LINE:
1. Convert video to frames and rotate the frames vertically.
2. Using 5 point mask, normalize the face(highlight the prominent features) descarding the background.
3. On top of the Aligned face, apply a mesh with 68 landmarks.
4. Just super-imposing the landmarks on the video/frames, will lead to unstable alginmet of landmark points on the video. We have to stabilize the landmarks using LK method to track the points in the current frame w.r.t. previous frame. Then we take a weighted average of the two measurements and that is the stabilized landmarks point. 

4. Combined the videos from different stages, one the original video, two after applying 68 point landmarks, third after stabilizing the landmarks.
