import edu.wpi.first.vision.*;
import edu.wpi.cscore.CvSource;
import edu.wpi.first.networktables.*;

import java.util.Random;

import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgproc.Moments;

public class GripListener implements VisionRunner.Listener<GripPipeline> {

    /** The vision network table. */
    public static final String TABLE_NAME = "Vision";

    /** Horizontal angle in degrees to the target. */
    public static final String TARGET_ANGLE_X = "target.angleX";

    /** Vertical angle in degrees to the target. */
    public static final String TARGET_ANGLE_Y = "target.angleY";

    /** Distanct to the target in inches. */
    public static final String TARGET_DISTANCE = "target.distance";

    /** Confidence that we see a valid target, in the range 0.0 to 1.0. */
    public static final String TARGET_CONFIDENCE = "target.confidence";

    /** Processing throughput in frames per second. */
    public static final String TARGET_FPS = "target.fps";

    private final NetworkTableInstance ntinst;
    private final NetworkTable networkTable;
    private final CvSource targetStream;
    private long previousTime;

    // Camera FOV, change if switching cameras
    private static final int fovx = 40;
    private static final int fovy = 30;

    public GripListener(NetworkTableInstance nti, CvSource stream) {
        ntinst = nti;
        targetStream = stream;
        networkTable = ntinst.getTable(TABLE_NAME);
        previousTime = System.currentTimeMillis();
    }

    @Override
    public void copyPipelineOutputs(GripPipeline pipeline) {
        double angleX = 0.0;
        double angleY = 0.0;
        double distance = 0.0;
        double confidence = 0.0;

        // Mat image = pipeline.hsvThresholdOutput();
        Mat image = new Mat(pipeline.hsvThresholdOutput().rows(), pipeline.hsvThresholdOutput().cols(), CvType.CV_8UC3);

        for (int i = 0; i < pipeline.convexHullsOutput().size(); i++) {
            Tilt t = getHullTilt(pipeline.convexHullsOutput().get(i));
            if(t == Tilt.Left){
               Imgproc.drawContours(image, pipeline.convexHullsOutput(), i, new Scalar(0, 0, 255), 5); 
            }else{
                Imgproc.drawContours(image, pipeline.convexHullsOutput(), i, new Scalar(0, 255, 0), 5);
            }
            
        }

        if (pipeline.convexHullsOutput().size() > 0) {
            MatOfPoint largestHull = pipeline.convexHullsOutput().get(0);
            for (int i = 0; i < pipeline.convexHullsOutput().size(); ++i) {
                if (Imgproc.contourArea(largestHull) < Imgproc.contourArea(pipeline.convexHullsOutput().get(i))) {
                    largestHull = pipeline.convexHullsOutput().get(i);
                }
            }

            // Finds center of largestHull
            Point center = centerOfConvexHull(largestHull);

            // Draws circle on center
            // Imgproc.circle(image, center, 10, new Scalar(255, 0, 0), 10);

            // Sets angleX and angleY
            angleX = findAngle(center.x, image.cols(), fovx);
            angleY = findAngle(center.y, image.rows(), fovy);
        }
        // TODO : everything

        long timeSpan = System.currentTimeMillis() - previousTime;
        previousTime = System.currentTimeMillis();
        networkTable.getEntry(TARGET_FPS).setNumber(Math.round(1000.0 / timeSpan));
        networkTable.getEntry(TARGET_ANGLE_X).setNumber(angleX);
        networkTable.getEntry(TARGET_ANGLE_Y).setNumber(angleY);
        networkTable.getEntry(TARGET_DISTANCE).setNumber(distance);
        networkTable.getEntry(TARGET_CONFIDENCE).setNumber(confidence);
        ntinst.flush();
        targetStream.putFrame(image);
    }

    // Finds center of convex hull
    public static Point centerOfConvexHull(MatOfPoint hull) {
        Moments moment = Imgproc.moments(hull);
        Point center = new Point();
        center.x = (int) (moment.get_m10() / moment.get_m00());
        center.y = (int) (moment.get_m01() / moment.get_m00());
        return center;
    }

    // Find angle from pixel data
    public static double findAngle(double pixel, int resolution, int fov) {
        double center = pixel - (resolution / 2);
        double fovtoradians = (Math.PI / 180) * fov;
        double ratio = center * (Math.sin(.5 * fovtoradians) / (.5 * resolution));
        double radians = Math.asin(ratio);
        double out = (180 / Math.PI) * radians;
        return out;
    }

    enum Tilt {
        Left, Right;
    }

    public static Tilt getHullTilt(MatOfPoint hull) {
        Mat line = new Mat();
        Imgproc.fitLine(hull, line, Imgproc.CV_DIST_L2, 0, 0.1, 0.1);
        if (line.get(1, 0)[0] > 0.0) {
            return Tilt.Left;
        } else {
            return Tilt.Right;
        }
    }

}