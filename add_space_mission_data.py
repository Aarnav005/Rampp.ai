import weaviate
import weaviate.classes as wvc
from weaviate.exceptions import WeaviateConnectionError

try:
    # Connect to local Weaviate instance
    client = weaviate.connect_to_local(
        host="localhost",
        port=8080,
        grpc_port=50051
    )

    if not client.is_ready():
        raise WeaviateConnectionError("Weaviate is not ready")

    print("Connected to Weaviate successfully.")

    # Delete existing SpaceMission collection if it exists
    collection_name = "SpaceMission"
    if client.collections.exists(collection_name):
        print(f"Collection '{collection_name}' already exists. Deleting it.")
        client.collections.delete(collection_name)

    # Create new SpaceMission collection
    print(f"Creating collection '{collection_name}'.")
    mission_collection = client.collections.create(
        name="SpaceMission",
        description="Information about space exploration missions.",
        vectorizer_config=wvc.config.Configure.Vectorizer.text2vec_transformers(),
        properties=[
            wvc.config.Property(name="mission_name", data_type=wvc.config.DataType.TEXT),
            wvc.config.Property(name="launch_date", data_type=wvc.config.DataType.DATE),
            wvc.config.Property(name="agency", data_type=wvc.config.DataType.TEXT),
            wvc.config.Property(name="mission_type", data_type=wvc.config.DataType.TEXT),
            wvc.config.Property(name="objective", data_type=wvc.config.DataType.TEXT),
            wvc.config.Property(name="status", data_type=wvc.config.DataType.TEXT),
            wvc.config.Property(name="destination", data_type=wvc.config.DataType.TEXT),
        ]
    )

    print("Collection 'SpaceMission' created successfully.")

    # Comprehensive sample data with 55 missions total
    missions = [
        # Original 5 missions
        {
            "mission_name": "Apollo 11",
            "launch_date": "1969-07-16T00:00:00Z",
            "agency": "NASA",
            "mission_type": "Manned",
            "objective": "First human landing on the Moon",
            "status": "Success",
            "destination": "Moon"
        },
        {
            "mission_name": "Mars Pathfinder",
            "launch_date": "1996-12-04T00:00:00Z",
            "agency": "NASA",
            "mission_type": "Rover",
            "objective": "Explore Martian surface and test new landing technology",
            "status": "Success",
            "destination": "Mars"
        },
        {
            "mission_name": "Chandrayaan-2",
            "launch_date": "2019-07-22T00:00:00Z",
            "agency": "ISRO",
            "mission_type": "Orbiter + Lander",
            "objective": "Explore lunar south pole",
            "status": "Partial Success",
            "destination": "Moon"
        },
        {
            "mission_name": "James Webb Space Telescope",
            "launch_date": "2021-12-25T00:00:00Z",
            "agency": "NASA/ESA/CSA",
            "mission_type": "Observatory",
            "objective": "Deep space infrared observation of early universe",
            "status": "Ongoing",
            "destination": "L2 Orbit"
        },
        {
            "mission_name": "Artemis I",
            "launch_date": "2022-11-16T00:00:00Z",
            "agency": "NASA",
            "mission_type": "Unmanned",
            "objective": "Test Orion spacecraft and SLS rocket for future crewed missions",
            "status": "Success",
            "destination": "Moon"
        },
        
        # Additional 50 missions with common field values for testing
        
        # More Apollo missions (Common: NASA, Moon, Manned/Unmanned)
        {
            "mission_name": "Apollo 8",
            "launch_date": "1968-12-21T00:00:00Z",
            "agency": "NASA",
            "mission_type": "Manned",
            "objective": "First crewed mission to orbit the Moon",
            "status": "Success",
            "destination": "Moon"
        },
        {
            "mission_name": "Apollo 12",
            "launch_date": "1969-11-14T00:00:00Z",
            "agency": "NASA",
            "mission_type": "Manned",
            "objective": "Second human lunar landing mission",
            "status": "Success",
            "destination": "Moon"
        },
        {
            "mission_name": "Apollo 15",
            "launch_date": "1971-07-26T00:00:00Z",
            "agency": "NASA",
            "mission_type": "Manned",
            "objective": "Extended lunar surface exploration with lunar rover",
            "status": "Success",
            "destination": "Moon"
        },
        {
            "mission_name": "Apollo 17",
            "launch_date": "1972-12-07T00:00:00Z",
            "agency": "NASA",
            "mission_type": "Manned",
            "objective": "Final Apollo lunar landing mission",
            "status": "Success",
            "destination": "Moon"
        },
        
        # Mars missions (Common: Mars destination, various agencies)
        {
            "mission_name": "Mars Global Surveyor",
            "launch_date": "1996-11-07T00:00:00Z",
            "agency": "NASA",
            "mission_type": "Orbiter",
            "objective": "Global mapping and climate monitoring of Mars",
            "status": "Success",
            "destination": "Mars"
        },
        {
            "mission_name": "Mars Exploration Rover Spirit",
            "launch_date": "2003-06-10T00:00:00Z",
            "agency": "NASA",
            "mission_type": "Rover",
            "objective": "Search for evidence of past water activity on Mars",
            "status": "Success",
            "destination": "Mars"
        },
        {
            "mission_name": "Mars Exploration Rover Opportunity",
            "launch_date": "2003-07-07T00:00:00Z",
            "agency": "NASA",
            "mission_type": "Rover",
            "objective": "Search for evidence of past water activity on Mars",
            "status": "Success",
            "destination": "Mars"
        },
        {
            "mission_name": "Mars Reconnaissance Orbiter",
            "launch_date": "2005-08-12T00:00:00Z",
            "agency": "NASA",
            "mission_type": "Orbiter",
            "objective": "High-resolution imaging and atmospheric analysis of Mars",
            "status": "Ongoing",
            "destination": "Mars"
        },
        {
            "mission_name": "Curiosity Mars Rover",
            "launch_date": "2011-11-26T00:00:00Z",
            "agency": "NASA",
            "mission_type": "Rover",
            "objective": "Assess Mars' past habitability potential",
            "status": "Ongoing",
            "destination": "Mars"
        },
        {
            "mission_name": "Perseverance Mars Rover",
            "launch_date": "2020-07-30T00:00:00Z",
            "agency": "NASA",
            "mission_type": "Rover",
            "objective": "Search for signs of ancient microbial life on Mars",
            "status": "Ongoing",
            "destination": "Mars"
        },
        {
            "mission_name": "Mars Orbiter Mission (Mangalyaan)",
            "launch_date": "2013-11-05T00:00:00Z",
            "agency": "ISRO",
            "mission_type": "Orbiter",
            "objective": "Study Martian atmosphere and surface features",
            "status": "Success",
            "destination": "Mars"
        },
        {
            "mission_name": "ExoMars Trace Gas Orbiter",
            "launch_date": "2016-03-14T00:00:00Z",
            "agency": "ESA/Roscosmos",
            "mission_type": "Orbiter",
            "objective": "Study trace gases in Martian atmosphere",
            "status": "Ongoing",
            "destination": "Mars"
        },
        
        # ISRO missions (Common agency)
        {
            "mission_name": "Chandrayaan-1",
            "launch_date": "2008-10-22T00:00:00Z",
            "agency": "ISRO",
            "mission_type": "Orbiter",
            "objective": "Map lunar surface and search for water ice",
            "status": "Success",
            "destination": "Moon"
        },
        {
            "mission_name": "Chandrayaan-3",
            "launch_date": "2023-07-14T00:00:00Z",
            "agency": "ISRO",
            "mission_type": "Lander + Rover",
            "objective": "Soft landing on lunar south pole",
            "status": "Success",
            "destination": "Moon"
        },
        {
            "mission_name": "Aditya-L1",
            "launch_date": "2023-09-02T00:00:00Z",
            "agency": "ISRO",
            "mission_type": "Observatory",
            "objective": "Study solar corona and solar wind",
            "status": "Ongoing",
            "destination": "L1 Orbit"
        },
        
        # ESA missions (Common agency)
        {
            "mission_name": "Rosetta",
            "launch_date": "2004-03-02T00:00:00Z",
            "agency": "ESA",
            "mission_type": "Orbiter + Lander",
            "objective": "Study comet 67P/Churyumov-Gerasimenko",
            "status": "Success",
            "destination": "Comet 67P"
        },
        {
            "mission_name": "BepiColombo",
            "launch_date": "2018-10-20T00:00:00Z",
            "agency": "ESA/JAXA",
            "mission_type": "Orbiter",
            "objective": "Study Mercury's composition and magnetic field",
            "status": "Ongoing",
            "destination": "Mercury"
        },
        {
            "mission_name": "Solar Orbiter",
            "launch_date": "2020-02-10T00:00:00Z",
            "agency": "ESA/NASA",
            "mission_type": "Observatory",
            "objective": "Study solar wind and solar magnetic field",
            "status": "Ongoing",
            "destination": "Solar Orbit"
        },
        
        # Observatory missions (Common mission type)
        {
            "mission_name": "Hubble Space Telescope",
            "launch_date": "1990-04-24T00:00:00Z",
            "agency": "NASA/ESA",
            "mission_type": "Observatory",
            "objective": "Deep space optical and UV observations",
            "status": "Ongoing",
            "destination": "Earth Orbit"
        },
        {
            "mission_name": "Spitzer Space Telescope",
            "launch_date": "2003-08-25T00:00:00Z",
            "agency": "NASA",
            "mission_type": "Observatory",
            "objective": "Infrared astronomical observations",
            "status": "Success",
            "destination": "Solar Orbit"
        },
        {
            "mission_name": "Kepler Space Telescope",
            "launch_date": "2009-03-07T00:00:00Z",
            "agency": "NASA",
            "mission_type": "Observatory",
            "objective": "Search for Earth-size exoplanets",
            "status": "Success",
            "destination": "Solar Orbit"
        },
        {
            "mission_name": "TESS",
            "launch_date": "2018-04-18T00:00:00Z",
            "agency": "NASA",
            "mission_type": "Observatory",
            "objective": "Search for exoplanets around nearby stars",
            "status": "Ongoing",
            "destination": "L2 Orbit"
        },
        
        # Venus missions (Common destination)
        {
            "mission_name": "Magellan",
            "launch_date": "1989-05-04T00:00:00Z",
            "agency": "NASA",
            "mission_type": "Orbiter",
            "objective": "Radar mapping of Venus surface",
            "status": "Success",
            "destination": "Venus"
        },
        {
            "mission_name": "Venus Express",
            "launch_date": "2005-11-09T00:00:00Z",
            "agency": "ESA",
            "mission_type": "Orbiter",
            "objective": "Study Venus atmosphere and surface",
            "status": "Success",
            "destination": "Venus"
        },
        {
            "mission_name": "Akatsuki",
            "launch_date": "2010-05-21T00:00:00Z",
            "agency": "JAXA",
            "mission_type": "Orbiter",
            "objective": "Study Venus meteorology and atmosphere",
            "status": "Ongoing",
            "destination": "Venus"
        },
        
        # Jupiter missions (Common destination)
        {
            "mission_name": "Galileo",
            "launch_date": "1989-10-18T00:00:00Z",
            "agency": "NASA",
            "mission_type": "Orbiter + Probe",
            "objective": "Study Jupiter and its moons",
            "status": "Success",
            "destination": "Jupiter"
        },
        {
            "mission_name": "Juno",
            "launch_date": "2011-08-05T00:00:00Z",
            "agency": "NASA",
            "mission_type": "Orbiter",
            "objective": "Study Jupiter's composition and magnetic field",
            "status": "Ongoing",
            "destination": "Jupiter"
        },
        {
            "mission_name": "JUICE",
            "launch_date": "2023-04-14T00:00:00Z",
            "agency": "ESA",
            "mission_type": "Orbiter",
            "objective": "Study Jupiter's icy moons",
            "status": "Ongoing",
            "destination": "Jupiter"
        },
        
        # Saturn missions
        {
            "mission_name": "Cassini-Huygens",
            "launch_date": "1997-10-15T00:00:00Z",
            "agency": "NASA/ESA/ASI",
            "mission_type": "Orbiter + Lander",
            "objective": "Study Saturn system and land on Titan",
            "status": "Success",
            "destination": "Saturn"
        },
        
        # Failed missions (Common status)
        {
            "mission_name": "Mars Climate Orbiter",
            "launch_date": "1998-12-11T00:00:00Z",
            "agency": "NASA",
            "mission_type": "Orbiter",
            "objective": "Study Martian climate and atmosphere",
            "status": "Failed",
            "destination": "Mars"
        },
        {
            "mission_name": "Beagle 2",
            "launch_date": "2003-06-02T00:00:00Z",
            "agency": "ESA/UK",
            "mission_type": "Lander",
            "objective": "Search for signs of life on Mars",
            "status": "Failed",
            "destination": "Mars"
        },
        {
            "mission_name": "Phobos-Grunt",
            "launch_date": "2011-11-09T00:00:00Z",
            "agency": "Roscosmos",
            "mission_type": "Sample Return",
            "objective": "Return samples from Phobos moon of Mars",
            "status": "Failed",
            "destination": "Mars"
        },
        
        # Chinese missions (Common agency)
        {
            "mission_name": "Chang'e-3",
            "launch_date": "2013-12-01T00:00:00Z",
            "agency": "CNSA",
            "mission_type": "Lander + Rover",
            "objective": "Soft landing on Moon with Yutu rover",
            "status": "Success",
            "destination": "Moon"
        },
        {
            "mission_name": "Chang'e-4",
            "launch_date": "2018-12-07T00:00:00Z",
            "agency": "CNSA",
            "mission_type": "Lander + Rover",
            "objective": "First soft landing on Moon's far side",
            "status": "Success",
            "destination": "Moon"
        },
        {
            "mission_name": "Chang'e-5",
            "launch_date": "2020-11-23T00:00:00Z",
            "agency": "CNSA",
            "mission_type": "Sample Return",
            "objective": "Return lunar samples to Earth",
            "status": "Success",
            "destination": "Moon"
        },
        {
            "mission_name": "Tianwen-1",
            "launch_date": "2020-07-23T00:00:00Z",
            "agency": "CNSA",
            "mission_type": "Orbiter + Rover",
            "objective": "Study Mars surface and atmosphere",
            "status": "Ongoing",
            "destination": "Mars"
        },
        
        # Japanese missions (JAXA)
        {
            "mission_name": "Hayabusa",
            "launch_date": "2003-05-09T00:00:00Z",
            "agency": "JAXA",
            "mission_type": "Sample Return",
            "objective": "Return samples from asteroid Itokawa",
            "status": "Success",
            "destination": "Asteroid Itokawa"
        },
        {
            "mission_name": "Hayabusa2",
            "launch_date": "2014-12-03T00:00:00Z",
            "agency": "JAXA",
            "mission_type": "Sample Return",
            "objective": "Return samples from asteroid Ryugu",
            "status": "Success",
            "destination": "Asteroid Ryugu"
        },
        {
            "mission_name": "SLIM",
            "launch_date": "2023-09-07T00:00:00Z",
            "agency": "JAXA",
            "mission_type": "Lander",
            "objective": "Demonstrate precision lunar landing technology",
            "status": "Partial Success",
            "destination": "Moon"
        },
        
        # Private company missions
        {
            "mission_name": "Falcon Heavy Demo",
            "launch_date": "2018-02-06T00:00:00Z",
            "agency": "SpaceX",
            "mission_type": "Test Flight",
            "objective": "Demonstrate Falcon Heavy capabilities",
            "status": "Success",
            "destination": "Solar Orbit"
        },
        {
            "mission_name": "Beresheet",
            "launch_date": "2019-02-21T00:00:00Z",
            "agency": "SpaceIL",
            "mission_type": "Lander",
            "objective": "First private lunar landing attempt",
            "status": "Failed",
            "destination": "Moon"
        },
        {
            "mission_name": "CAPSTONE",
            "launch_date": "2022-06-28T00:00:00Z",
            "agency": "NASA/Advanced Space",
            "mission_type": "CubeSat",
            "objective": "Test lunar Gateway orbit dynamics",
            "status": "Success",
            "destination": "Moon"
        },
        
        # Outer solar system missions
        {
            "mission_name": "Voyager 1",
            "launch_date": "1977-09-05T00:00:00Z",
            "agency": "NASA",
            "mission_type": "Flyby",
            "objective": "Grand tour of outer planets",
            "status": "Ongoing",
            "destination": "Interstellar Space"
        },
        {
            "mission_name": "Voyager 2",
            "launch_date": "1977-08-20T00:00:00Z",
            "agency": "NASA",
            "mission_type": "Flyby",
            "objective": "Grand tour of outer planets",
            "status": "Ongoing",
            "destination": "Interstellar Space"
        },
        {
            "mission_name": "New Horizons",
            "launch_date": "2006-01-19T00:00:00Z",
            "agency": "NASA",
            "mission_type": "Flyby",
            "objective": "Study Pluto and Kuiper Belt objects",
            "status": "Ongoing",
            "destination": "Kuiper Belt"
        },
        
        # Recent/upcoming missions
        {
            "mission_name": "DART",
            "launch_date": "2021-11-24T00:00:00Z",
            "agency": "NASA",
            "mission_type": "Impactor",
            "objective": "Test planetary defense by impacting asteroid",
            "status": "Success",
            "destination": "Asteroid Dimorphos"
        },
        {
            "mission_name": "Lucy",
            "launch_date": "2021-10-16T00:00:00Z",
            "agency": "NASA",
            "mission_type": "Flyby",
            "objective": "Study Jupiter Trojan asteroids",
            "status": "Ongoing",
            "destination": "Jupiter Trojans"
        },
        {
            "mission_name": "OSIRIS-REx",
            "launch_date": "2016-09-08T00:00:00Z",
            "agency": "NASA",
            "mission_type": "Sample Return",
            "objective": "Return samples from asteroid Bennu",
            "status": "Success",
            "destination": "Asteroid Bennu"
        },
        {
            "mission_name": "Parker Solar Probe",
            "launch_date": "2018-08-12T00:00:00Z",
            "agency": "NASA",
            "mission_type": "Probe",
            "objective": "Study solar corona by flying close to Sun",
            "status": "Ongoing",
            "destination": "Solar Corona"
        },
        {
            "mission_name": "Artemis II",
            "launch_date": "2025-11-01T00:00:00Z",
            "agency": "NASA",
            "mission_type": "Manned",
            "objective": "First crewed mission around Moon since Apollo",
            "status": "Planned",
            "destination": "Moon"
        }
    ]


    # Insert missions in batches
    batch_size = 10
    for i in range(0, len(missions), batch_size):
        batch = missions[i:i+batch_size]
        mission_collection.data.insert_many(batch)
        print(f"Inserted missions {i+1} to {i+len(batch)}")

    print("All missions inserted successfully.")

except Exception as e:
    print(f"An error occurred: {e}")
finally:
    if 'client' in locals():
        client.close()
        print("Weaviate client connection closed.") 