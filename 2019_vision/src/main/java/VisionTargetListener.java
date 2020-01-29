import java.io.File;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgproc.Moments;

import edu.wpi.cscore.CvSource;
import edu.wpi.first.networktables.NetworkTable;
import edu.wpi.first.networktables.NetworkTableInstance;
import edu.wpi.first.vision.VisionRunner;

public class VisionTargetListener implements VisionRunner.Listener<VisionTargetPipeline> {

    /** The vision network table. */
    public static final String TABLE_NAME = "visionTarget";

    /** Horizontal angle in degrees to the target. */
    public static final String TARGET_ANGLE_X = "visionTarget.angleX";

    /** Vertical angle in degrees to the target. */
    public static final String TARGET_ANGLE_Y = "visionTarget.angleY";

    /** Distanct to the target in inches. */
    public static final String TARGET_DISTANCE = "visionTarget.distance";

    /** Confidence that we see a valid target, in the range 0.0 to 1.0. */
    public static final String TARGET_CONFIDENCE = "visionTarget.confidence";

    /** Number of vision target pairs */
    public static final String TARGET_PAIRS = "visionTarget.pairs";

    /** Processing throughput in frames per second. */
    public static final String TARGET_FPS = "visionTarget.fps";

    /** Milliseconds between saving image files.  A negative value indicates to not save. */
    public static final String TARGET_SAVE = "visionTarget.saveImageTime";

    /** Width of the best image pair, measured in pixels. */
    public static final String TARGET_WIDTH = "visionTarget.width";

    private final NetworkTableInstance ntinst;
    private final NetworkTable networkTable;
    private final CvSource targetStream;
    private long previousTime;

    // Camera FOV, change if switching cameras
    private static final int fovx = 40;
    private static final int fovy = 30;

    final int referenceDist = 36;
    final int referenceWidth = 64;
    final double referenceTargetWidth = 11.25;

    final double focalLength = referenceWidth * referenceDist / referenceTargetWidth;

    final SimpleDateFormat imageDateFormat;
    private long saveImageTimeout = 0;
    private long prevSaveImageTimeout = 0;

    public VisionTargetListener(NetworkTableInstance nti, CvSource stream) {
        ntinst = nti;
        targetStream = stream;
        networkTable = ntinst.getTable(TABLE_NAME);
        previousTime = System.currentTimeMillis();
        imageDateFormat = new SimpleDateFormat("yyyyMMdd_HHmmss_SSS");
    }

    @Override
    public void copyPipelineOutputs(VisionTargetPipeline pipeline) {
        double angleX = 0.0;
        double angleY = 0.0;
        double distance = 0.0;
        double confidence = 0.0;

        // ArrayLists for left and right targets
        ArrayList<MatOfPoint> leftTargets = new ArrayList<MatOfPoint>();
        ArrayList<MatOfPoint> rightTargets = new ArrayList<MatOfPoint>();

        // Mat image = pipeline.hsvThresholdOutput();
        Mat image = new Mat(pipeline.hsvThresholdOutput().rows(), pipeline.hsvThresholdOutput().cols(), CvType.CV_8UC3);

        // draws contours in red and green, and adds convex hulls to left and right
        // ArrayLists
        for (int i = 0; i < pipeline.convexHullsOutput().size(); i++) {
            Tilt t = getHullTilt(pipeline.convexHullsOutput().get(i));
            if (t == Tilt.Left) {
                Imgproc.drawContours(image, pipeline.convexHullsOutput(), i, new Scalar(0, 0, 255), 3);
                leftTargets.add(pipeline.convexHullsOutput().get(i));
            } else {
                Imgproc.drawContours(image, pipeline.convexHullsOutput(), i, new Scalar(0, 255, 0), 3);
                rightTargets.add(pipeline.convexHullsOutput().get(i));
            }
        }

        // draws a blue rectangle arround paired targets
        ArrayList<TargetPair> targetPairs = new ArrayList<TargetPair>();
        for (int i = 0; i < leftTargets.size(); i++) {
            TargetPair pair = new TargetPair(leftTargets.get(i), rightTargets);
            if (pair.getHasPair()) {
                targetPairs.add(pair);
                Imgproc.rectangle(image, pair.topLeft(), pair.bottomRight(), new Scalar(255, 20, 10), -0);
                // Imgproc.circle(image, pairCenter, 12, new Scalar(255, 6, 6));
            }
        }

        // selects the "best pair" and draws a yellow rectangle
        double bestPairWidth = 0;
        if (targetPairs.size() > 0) {
            TargetPair bestPair = targetPairs.get(0);
            for (int i = 0; i < targetPairs.size(); ++i) {
                if (targetPairs.get(i).pairSpread() < bestPair.pairSpread()) {
                    bestPair = targetPairs.get(i);
                }
            }
            Imgproc.rectangle(image, bestPair.topLeft(), bestPair.bottomRight(), new Scalar(0, 255, 255), 2);
            // Sets angleX and angleY
            angleX = findAngle(bestPair.findCenter().x, image.cols(), fovx);
            angleY = findAngle(bestPair.findCenter().y, image.rows(), fovy);
            bestPairWidth = bestPair.pairSpread();
            distance = referenceTargetWidth * focalLength / bestPairWidth;

            if (pipeline.convexHullsOutput().size() > 0) {
                confidence = 1;
            } else {
                confidence = 0;
            }
        }

        long timeSpan = System.currentTimeMillis() - previousTime;
        previousTime = System.currentTimeMillis();
        networkTable.getEntry(TARGET_FPS).setNumber(Math.round(1000.0 / timeSpan));
        networkTable.getEntry(TARGET_ANGLE_X).setNumber(angleX);
        networkTable.getEntry(TARGET_ANGLE_Y).setNumber(angleY);
        networkTable.getEntry(TARGET_DISTANCE).setNumber(distance);
        networkTable.getEntry(TARGET_CONFIDENCE).setNumber(confidence);
        networkTable.getEntry(TARGET_PAIRS).setNumber(targetPairs.size());
        networkTable.getEntry(TARGET_WIDTH).setNumber(bestPairWidth);
        ntinst.flush();
        targetStream.putFrame(image);

        long saveImageTime = ((Number)networkTable.getEntry(TARGET_SAVE).getNumber(Double.valueOf(-1.0))).longValue();
        if (saveImageTime <= 0) {
            saveImageTimeout = 0;
        } else if (prevSaveImageTimeout <= 0) {
            saveImageFile(image);
            saveImageTimeout = System.currentTimeMillis() + saveImageTime;
        } else if (System.currentTimeMillis() > saveImageTimeout)  {
            saveImageFile(image);
            saveImageTimeout = System.currentTimeMillis() + saveImageTime;
        }
        prevSaveImageTimeout = saveImageTime;
    }

    /** Save an image matrix to a file. */
    protected File saveImageFile(Mat image) {
        try {
            File imageDir = findValidDirectory("/media/usb", "/media/usb0", "/media/usb1",
                    System.getProperty("user.home"), "/tmp");
            if (imageDir == null) {
                System.out.println("::: saveImageFile: FAIL");
                return null;
            }
            File imageFile = new File(imageDir, "visionTarget_" + imageDateFormat.format(new Date()) + ".jpg");
            boolean success = Imgcodecs.imwrite(imageFile.getAbsolutePath(), image);
            System.out.println("::: saveImageFile: " + success + " : " + imageFile.getAbsolutePath());
            return success ? imageFile : null;
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }

    /** Find the first directory in the list that exists and can be written to. */
    protected File findValidDirectory(String... dirNames) {
        for (String fileName : dirNames) {
            File dirName = new File(fileName);
            if (dirName.isDirectory() && dirName.canWrite()) {
                return dirName;
            }
        }
        return null;
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

    private class TargetPair {
        Point leftCenter;
        Point rightCenter;
        boolean hasPair = false;

        // pairs right and left convex hulls
        TargetPair(MatOfPoint left, ArrayList<MatOfPoint> rightMats) {
            leftCenter = centerOfConvexHull(left);
            Point bestFit = new Point(99999, 0);
            for (int i = 0; i < rightMats.size(); i++) {
                Point tmpRightCenter = centerOfConvexHull(rightMats.get(i));
                if (tmpRightCenter.x > leftCenter.x && tmpRightCenter.x < bestFit.x) {
                    bestFit = tmpRightCenter;
                    hasPair = true;
                }
            }
            rightCenter = bestFit;
        }

        public boolean getHasPair() {
            return hasPair;
        }

        // finds the center point of a target pair
        public Point findCenter() {
            double averageX = (rightCenter.x + leftCenter.x) / 2;
            double averageY = (rightCenter.y + leftCenter.y) / 2;
            return new Point(averageX, averageY);
        }

        public Point topLeft() {
            return new Point(leftCenter.x - 27, leftCenter.y - 37);
        }

        public Point bottomRight() {
            return new Point(rightCenter.x + 27, rightCenter.y + 37);
        }

        public double pairSpread() {
            return rightCenter.x - leftCenter.x;
        }
    }

    enum Tilt {
        Left, Right;
    }

    public static Tilt getHullTilt(MatOfPoint hull) {
        Mat line = new Mat();
        Imgproc.fitLine(hull, line, Imgproc.CV_DIST_L2, 0, 0.1, 0.1);
        if (line.get(1, 0)[0] < 0.0) {
            return Tilt.Left;
        } else {
            return Tilt.Right;
        }
    }
}