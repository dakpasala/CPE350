# incidentDetector.py
import math

def detect_incident(active_objects):
    incidents = []
    objs = list(active_objects.values())

    for i in range(len(objs)):
        for j in range(i+1, len(objs)):
            A, B = objs[i], objs[j]
            dist = math.dist(A.position, B.position)

            # Near collision
            if dist < 2.5:
                rel_speed = abs(A.speed - B.speed)
                heading_diff = abs(A.heading - B.heading)
                if heading_diff > 90 and rel_speed > 5:
                    incidents.append(("near_miss", A.id, B.id))

        # Sudden stop
        if len(A.speed_history) >= 2:
            dv = A.speed_history[-1] - A.speed_history[-2]
            if dv < -8:  # m/s² threshold
                incidents.append(("sudden_stop", A.id))

    return incidents
