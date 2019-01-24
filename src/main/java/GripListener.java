import edu.wpi.first.vision.*;
import edu.wpi.first.networktables.*;

import java.util.Random;

import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

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
    private long previousTime;

    public GripListener(NetworkTableInstance nti) {
        ntinst = nti;
        networkTable = ntinst.getTable(TABLE_NAME);
        previousTime = System.currentTimeMillis();
    }

    @Override
    public void copyPipelineOutputs(GripPipeline pipeline) {
        double angleX = 0.0;
        double angleY = 0.0;
        double distance = 0.0;
        double confidence = 0.0;


        // TODO : everything


        long timeSpan = System.currentTimeMillis() - previousTime;
        previousTime = System.currentTimeMillis();
        networkTable.getEntry(TARGET_FPS).setNumber(Math.round(1000.0 / timeSpan));
        networkTable.getEntry(TARGET_ANGLE_X).setNumber(angleX);
        networkTable.getEntry(TARGET_ANGLE_Y).setNumber(angleY);
        networkTable.getEntry(TARGET_DISTANCE).setNumber(distance);
        networkTable.getEntry(TARGET_CONFIDENCE).setNumber(confidence);
        ntinst.flush();
    }

}