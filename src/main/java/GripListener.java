import edu.wpi.first.vision.*;
import edu.wpi.first.networktables.*;

public class GripListener implements VisionRunner.Listener<GripPipeline> {

    private final NetworkTable networkTable;

    public GripListener(NetworkTableInstance ntinst) {
        networkTable = ntinst.getTable("Vision");
    }

    @Override
    public void copyPipelineOutputs(GripPipeline pipeline) {
        // TODO : look for vision targets.
        // TODO : report results into the network table.
    }

}