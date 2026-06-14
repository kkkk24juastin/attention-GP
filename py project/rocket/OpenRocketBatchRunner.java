import info.openrocket.core.document.OpenRocketDocument;
import info.openrocket.core.document.Simulation;
import info.openrocket.core.file.GeneralRocketLoader;
import info.openrocket.core.simulation.FlightData;
import info.openrocket.core.startup.OpenRocketCore;

import java.io.File;
import java.util.Locale;

public class OpenRocketBatchRunner {
    public static void main(String[] args) throws Exception {
        if (args.length < 1) {
            throw new IllegalArgumentException("Usage: OpenRocketBatchRunner <ork-file> [simulation-index]");
        }

        Locale.setDefault(Locale.US);
        File orkFile = new File(args[0]);
        int simulationIndex = args.length >= 2 ? Integer.parseInt(args[1]) : 0;

        OpenRocketCore.initialize();
        GeneralRocketLoader loader = new GeneralRocketLoader(orkFile);
        OpenRocketDocument document = loader.load();
        Simulation simulation = document.getSimulation(simulationIndex);
        simulation.simulate();

        FlightData data = simulation.getSimulatedData();
        System.out.printf(
                Locale.US,
                "OPENROCKET_RESULT\t%.10f\t%.10f\t%.10f\t%.10f\t%.10f%n",
                data.getMaxAltitude(),
                data.getFlightTime(),
                data.getTimeToApogee(),
                data.getMaxVelocity(),
                data.getMaxAcceleration()
        );
    }
}
