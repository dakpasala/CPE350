from collections import defaultdict

# Objective: within each five minute bin, push a heatmap containing latitude/longitude position frequencies summed up across the bin

'''
    heatmap should be a 2d hashmap
    granularity is the number of decimal places to make boxes
'''
def add_to_heatmap(heatmap, roadObjectData, granularity = 6):
    # for each coordinate in the map path:
    for coordinateSet in roadObjectData["mapPath"]:
        # get the latitude key
        latKey = str(round(coordinateSet[0], granularity))
        # get the longitude key
        lonKey = str(round(coordinateSet[1], granularity))
        # check that the latitude bin exists
        if latKey not in heatmap:
            heatmap[latKey] = {}
        # check that the longitude bin exists
        if lonKey not in heatmap[latKey]:
            heatmap[latKey][lonKey] = 0
        # increment the key
        heatmap[latKey][lonKey] += 1

# Take in the collection of heatmaps and then combine them into a mongo-db friendly format
def extract_heatmap(heatmap_collection):
    output_heatmap = {}
    for heatmap in heatmap_collection:
        for latKey in heatmap:
            for lonKey in heatmap[latKey]:
                output_key = f"{latKey}_{lonKey}"
                output_heatmap[output_key] = output_heatmap.get(output_key, 0) + heatmap[latKey][lonKey]
    output_array = []
    for key in output_heatmap:
        output_array.append({"coordinate": key, "weight": output_heatmap[key]})
    return output_array

if __name__ == "__main__":
    heatmapDict ={}
    objectData = {
        "mapPath": [
            [45.65239813827, -120.68724978425298], [45.13268798215845, -120.29845321659876584565], [45.1654687224, -120.56897351], [45.12389754465, -120.5689712654],
        ]
    }
    add_to_heatmap(heatmapDict, objectData)
    print(heatmapDict)
    print(extract_heatmap(heatmapDict))