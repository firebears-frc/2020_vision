import edu.wpi.first.vision.*;
import edu.wpi.cscore.CvSource;
import edu.wpi.first.networktables.*;
import java.util.ArrayList;
import java.util.Random;

import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgproc.Moments;

public class OrangeBallListener implements VisionRunner.Listener<OrangeBallPipeline> {

    /** The vision network table. */
    public static final String TABLE_NAME = "orangeBall";

    /** Horizontal angle in degrees to the target. */
    public static final String TARGET_ANGLE_X = "orangeBall.angleX";

    /** Vertical angle in degrees to the target. */
    public static final String TARGET_ANGLE_Y = "orangeBall.angleY";

    /** Distanct to the target in inches. */
    public static final String TARGET_DISTANCE = "orangeBall.distance";

    /** Confidence that we see a valid target, in the range 0.0 to 1.0. */
    public static final String TARGET_CONFIDENCE = "orangeBall.confidence";

    /** Number of vision target pairs */
    public static final String TARGET_OBJECTS = "orangeBall.objects";

    /** Processing throughput in frames per second. */
    public static final String TARGET_FPS = "orangeBall.fps";

    public static final String TARGET_WIDTH = "orangeBall.width";

    private final NetworkTableInstance ntinst;
    private final NetworkTable networkTable;
    private final CvSource targetStream;
    private long previousTime;

    // Camera FOV, change if switching cameras
    private static final int fovx = 40;
    private static final int fovy = 30;

    final int referenceDist = 36;
    final int referenceWidth = 225;
    final double referenceTargetWidth = 11.25;

    final double focalLength = referenceWidth * referenceDist / referenceTargetWidth;

    public OrangeBallListener(NetworkTableInstance nti, CvSource stream) {
        ntinst = nti;
        targetStream = stream;
        networkTable = ntinst.getTable(TABLE_NAME);
        previousTime = System.currentTimeMillis();
    }

    @Override
    public void copyPipelineOutputs(OrangeBallPipeline pipeline) {
        double angleX = 0.0;
        double angleY = 0.0;
        double distance = 0.0;
        double confidence = 0.0;
        double largestHullwidth = 0.0;

        // Mat image = pipeline.hsvThresholdOutput();
        Mat image = new Mat(pipeline.hsvThresholdOutput().rows(), pipeline.hsvThresholdOutput().cols(), CvType.CV_8UC3);

        // draws contours in green, and adds convex hulls to array list
        for (int i = 0; i < pipeline.convexHullsOutput().size(); i++) {
            Imgproc.drawContours(image, pipeline.convexHullsOutput(), i, new Scalar(0, 255, 0), 1);
        }

        // finds the largest hull and draws a yellow outline arround the hull
        if (pipeline.convexHullsOutput().size() > 0) {
            MatOfPoint largestHull = pipeline.convexHullsOutput().get(0);
            for (int i = 1; i < pipeline.convexHullsOutput().size(); ++i) {
                if (Imgproc.contourArea(largestHull) < Imgproc.contourArea(pipeline.convexHullsOutput().get(i))) {
                    largestHull = pipeline.convexHullsOutput().get(i);
                    Imgproc.drawContours(image, pipeline.convexHullsOutput(), i, new Scalar(10, 255, 255), 2);

                }
            }

            // Sets angleX, angleY and the pixel width of the largest hull
            angleX = findAngle(centerOfConvexHull(largestHull).x, image.cols(), fovx);
            angleY = findAngle(centerOfConvexHull(largestHull).y, image.rows(), fovy);
            largestHullwidth = Math.sqrt(Imgproc.contourArea(largestHull) / Math.PI) * 2;
            // finds the distance of the target in inches
            distance = referenceTargetWidth * focalLength / largestHullwidth;

            if (pipeline.convexHullsOutput().size() > 0) {
                confidence = 1;
            } else {
                confidence = 0;
            }
        }
        // TODO : everything

        long timeSpan = System.currentTimeMillis() - previousTime;
        previousTime = System.currentTimeMillis();
        networkTable.getEntry(TARGET_FPS).setNumber(Math.round(1000.0 / timeSpan));
        networkTable.getEntry(TARGET_ANGLE_X).setNumber(angleX);
        networkTable.getEntry(TARGET_ANGLE_Y).setNumber(angleY);
        networkTable.getEntry(TARGET_DISTANCE).setNumber(distance);
        networkTable.getEntry(TARGET_CONFIDENCE).setNumber(confidence);
        networkTable.getEntry(TARGET_OBJECTS).setNumber(pipeline.convexHullsOutput().size());
        networkTable.getEntry(TARGET_WIDTH).setNumber(largestHullwidth);
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
}