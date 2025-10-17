from matplotlib import path

def setLanes(pointSetsDict):
    lanes = {}
    for pointSetName in pointSetsDict:
        pointSet = pointSetsDict[pointSetName]
        pathPoints = [(pointSet[0][i], pointSet[1][i]) for i in range(0, len(pointSet[0]))]
        lanes[pointSetName] = path.Path(pathPoints)
    return lanes

def whichLane(coordinate: tuple[float], lanes):
    for lane in lanes:
        if lanes[lane].contains_points([coordinate]):
            return lane
    return "Unknown"

def setLanePairsFromDBList(dbLanes):
    laneCoords = {}
    for lane in dbLanes:
        coordsList = ([],[])
        for coordinate in lane["coordinates"]:
            coordsList[0].append(coordinate["lat"])
            coordsList[1].append(coordinate["lng"])
        laneCoords[lane['name']] = coordsList

    return setLanes(laneCoords)