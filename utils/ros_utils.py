import rospy
import tf2_geometry_msgs
from geometry_msgs.msg import PoseStamped

def transform_pose(pose:PoseStamped, transform, target_frame = None) -> PoseStamped:
    try:
        if (transform.child_frame_id != pose.header.frame_id):
            rospy.logwarn(f"source frame id from transform: {transform.child_frame_id} != from pose: {pose.header.frame_id}")
        transformed_pose = tf2_geometry_msgs.do_transform_pose(pose, transform)
        rospy.loginfo(f"Pose transformed from {pose.header.frame_id} to {transform.header.frame_id}")
        return transformed_pose
    except Exception as e:
        rospy.logerr(f"PoseStamped transformation failed: {e}")
        return pose