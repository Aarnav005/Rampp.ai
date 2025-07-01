import weaviate
import weaviate.classes as wvc
from weaviate.exceptions import WeaviateConnectionError
from datetime import datetime

def print_article_info(articles):
    """Helper function to print article information"""
    for i, article in enumerate(articles, 1):
        print(f"\n{i}. {article.properties['title']}")
        print(f"   By: {article.properties['author']}")
        print(f"   Category: {article.properties['category']}")

try:
    # Connect to Weaviate
    client = weaviate.connect_to_local(
        host="localhost",
        port=8080,  # Changed to match docker-compose.yml
        grpc_port=50051
    )

    # Verify connection
    if not client.is_ready():
        raise WeaviateConnectionError("Weaviate is not ready")

    print("Connected to Weaviate successfully")

    collection_name = "Article"
    if client.collections.exists(collection_name):
        print(f"Collection '{collection_name}' already exists. Deleting it for a fresh start.")
        client.collections.delete(collection_name)

    print(f"Creating collection '{collection_name}'.")
    articles_collection = client.collections.create(
        name="Article",
        description="A class representing a written article or document.",
        vectorizer_config=wvc.config.Configure.Vectorizer.text2vec_transformers(),
        properties=[
            wvc.config.Property(name="title", data_type=wvc.config.DataType.TEXT),
            wvc.config.Property(name="content", data_type=wvc.config.DataType.TEXT),
            wvc.config.Property(name="author", data_type=wvc.config.DataType.TEXT),
            wvc.config.Property(name="publish_date", data_type=wvc.config.DataType.DATE),
            wvc.config.Property(name="category", data_type=wvc.config.DataType.TEXT),
        ]
    )
    print(f"Collection '{collection_name}' created successfully.")

    # Sample articles data
    articles = [
        # Add these to your articles list in add_data_fixed.py
{"title": "AI Revolutionizes Healthcare", "content": "Artificial intelligence is transforming diagnostics and patient care.", "author": "Dr. Alice Smith", "publish_date": "2024-05-10T00:00:00Z", "category": "AI"},
{"title": "The Future of AI in Education", "content": "AI-powered tutors are personalizing learning for students worldwide.", "author": "Dr. Emily Zhang", "publish_date": "2024-11-15T00:00:00Z", "category": "AI"},
{"title": "Ethics in Artificial Intelligence", "content": "Debates on AI ethics are shaping global policy.", "author": "Arjun Mehta", "publish_date": "2025-01-20T00:00:00Z", "category": "AI"},
{"title": "AI and Climate Change", "content": "Machine learning models are helping predict climate patterns.", "author": "Dr. Sven Johansson", "publish_date": "2025-03-12T00:00:00Z", "category": "AI"},
{"title": "AI in Finance: Risks and Rewards", "content": "AI is disrupting traditional banking and investment.", "author": "Dr. Alice Smith", "publish_date": "2025-06-01T00:00:00Z", "category": "AI"},
# Technology (8 articles)
    {"title": "Edge Computing in Smart Cities", "content": "Processing data closer to sensors and devices is reducing latency and improving real-time urban services.", "author": "Arjun Mehta", "publish_date": "2025-08-05T00:00:00Z", "category": "Technology"},
    {"title": "Augmented Reality in Education", "content": "AR applications are creating immersive learning experiences in classrooms and remote education settings.", "author": "Dr. Emily Zhang", "publish_date": "2025-08-08T00:00:00Z", "category": "Technology"},
    {"title": "5G Network Infrastructure", "content": "Next-generation wireless networks are enabling new applications in autonomous vehicles and industrial IoT.", "author": "Lukas Schneider", "publish_date": "2025-08-11T00:00:00Z", "category": "Technology"},
    {"title": "Digital Twin Technology", "content": "Virtual replicas of physical systems are optimizing manufacturing processes and predictive maintenance.", "author": "Aisha Rahman", "publish_date": "2025-08-14T00:00:00Z", "category": "Technology"},
    {"title": "Neuromorphic Computing Chips", "content": "Brain-inspired processors are promising ultra-low power consumption for AI applications.", "author": "Dr. Victor Liao", "publish_date": "2025-08-17T00:00:00Z", "category": "Technology"},
    {"title": "Underwater Communication Networks", "content": "Acoustic and optical communication systems are connecting submarines and underwater research stations.", "author": "Arjun Mehta", "publish_date": "2025-08-20T00:00:00Z", "category": "Technology"},
    {"title": "Wearable Health Monitoring", "content": "Advanced sensors in clothing and accessories are providing continuous health tracking capabilities.", "author": "Dr. Emily Zhang", "publish_date": "2025-08-23T00:00:00Z", "category": "Technology"},
    {"title": "Quantum Internet Development", "content": "Quantum communication networks promise ultra-secure data transmission through quantum entanglement.", "author": "Lukas Schneider", "publish_date": "2025-08-26T00:00:00Z", "category": "Technology"},
    
    # Politics & Society (8 articles)
    {"title": "Smart Governance Initiatives", "content": "Digital platforms are streamlining government services and improving citizen access to public resources.", "author": "Miguel Alvarez", "publish_date": "2025-08-06T00:00:00Z", "category": "Politics"},
    {"title": "Rural Development Programs", "content": "Investment in rural infrastructure and technology is reducing urban-rural inequality gaps.", "author": "Dr. Elena Novak", "publish_date": "2025-08-09T00:00:00Z", "category": "Politics"},
    {"title": "Prison Reform Innovation", "content": "Rehabilitation programs and alternative sentencing are reducing recidivism rates and improving outcomes.", "author": "Khalid Musa", "publish_date": "2025-08-12T00:00:00Z", "category": "Politics"},
    {"title": "Youth Political Engagement", "content": "Young voters are using social media and grassroots organizing to influence policy and elections.", "author": "Sophia Lee", "publish_date": "2025-08-15T00:00:00Z", "category": "Politics"},
    {"title": "Diplomatic Technology Tools", "content": "Virtual reality and AI translation are enhancing international negotiations and cultural understanding.", "author": "Prof. Claire Dubois", "publish_date": "2025-08-18T00:00:00Z", "category": "Politics"},
    {"title": "Border Security Innovation", "content": "Advanced scanning technology and biometrics are improving security while facilitating legitimate travel.", "author": "Miguel Alvarez", "publish_date": "2025-08-21T00:00:00Z", "category": "Politics"},
    {"title": "Community Resilience Planning", "content": "Local governments are developing comprehensive disaster preparedness and emergency response strategies.", "author": "Dr. Elena Novak", "publish_date": "2025-08-24T00:00:00Z", "category": "Politics"},
    {"title": "Accessibility Rights Advocacy", "content": "Disability rights movements are pushing for universal design and inclusive technology in public spaces.", "author": "Khalid Musa", "publish_date": "2025-08-27T00:00:00Z", "category": "Politics"},
    
    # Health & Medicine (8 articles)
    {"title": "Immunotherapy Breakthroughs", "content": "Harnessing the immune system to fight cancer is showing remarkable success in clinical trials.", "author": "Dr. Miguel Santos", "publish_date": "2025-08-07T00:00:00Z", "category": "Health"},
    {"title": "Bioprinting Organ Development", "content": "3D printing of living tissues and organs is advancing toward solving organ shortage crises.", "author": "Prof. Hannah Kowalski", "publish_date": "2025-08-10T00:00:00Z", "category": "Health"},
    {"title": "Chronic Pain Management", "content": "New approaches combining medication, therapy, and technology are improving quality of life for pain sufferers.", "author": "Dr. Yusuf Khan", "publish_date": "2025-08-13T00:00:00Z", "category": "Health"},
    {"title": "Maternal Health Technology", "content": "Wearable devices and telemedicine are improving prenatal care access in underserved communities.", "author": "Dr. Clara Nguyen", "publish_date": "2025-08-16T00:00:00Z", "category": "Health"},
    {"title": "Artificial Limb Innovation", "content": "Mind-controlled prosthetics with sensory feedback are restoring function and sensation for amputees.", "author": "Dr. Kenji Tanaka", "publish_date": "2025-08-19T00:00:00Z", "category": "Health"},
    {"title": "Vaccine Development Platforms", "content": "mRNA technology and rapid development platforms are revolutionizing vaccine creation and distribution.", "author": "Dr. Miguel Santos", "publish_date": "2025-08-22T00:00:00Z", "category": "Health"},
    {"title": "Antimicrobial Resistance Solutions", "content": "New antibiotics and treatment strategies are addressing the growing threat of drug-resistant infections.", "author": "Prof. Hannah Kowalski", "publish_date": "2025-08-25T00:00:00Z", "category": "Health"},
    {"title": "Elder Care Technology", "content": "Smart home systems and monitoring devices are helping seniors age safely in their own homes.", "author": "Dr. Yusuf Khan", "publish_date": "2025-08-28T00:00:00Z", "category": "Health"},
    
    # Environment & Climate (8 articles)
    {"title": "Carbon Capture Scaling", "content": "Industrial-scale carbon capture and storage facilities are beginning to make meaningful climate impact.", "author": "Dr. Sven Johansson", "publish_date": "2025-08-29T00:00:00Z", "category": "Environment"},
    {"title": "Renewable Energy Integration", "content": "Smart grids and energy management systems are optimizing renewable energy distribution and storage.", "author": "Leila Hassan", "publish_date": "2025-08-30T00:00:00Z", "category": "Environment"},
    {"title": "Plastic Alternative Materials", "content": "Biodegradable polymers and packaging innovations are reducing plastic waste in consumer products.", "author": "Prof. Maria Petrova", "publish_date": "2025-08-31T00:00:00Z", "category": "Environment"},
    {"title": "Wildlife Corridor Creation", "content": "Connecting fragmented habitats with wildlife corridors is supporting animal migration and genetic diversity.", "author": "Ravi Singh", "publish_date": "2025-09-01T00:00:00Z", "category": "Environment"},
    {"title": "Extreme Weather Preparedness", "content": "Communities are developing early warning systems and infrastructure to withstand climate extremes.", "author": "Dr. Nina Feldman", "publish_date": "2025-09-02T00:00:00Z", "category": "Environment"},
    {"title": "Geothermal Energy Expansion", "content": "Enhanced geothermal systems are unlocking clean energy potential in previously unsuitable locations.", "author": "Dr. Sven Johansson", "publish_date": "2025-09-03T00:00:00Z", "category": "Environment"},
    {"title": "Soil Health Restoration", "content": "Regenerative agriculture practices are rebuilding soil carbon and improving land productivity.", "author": "Leila Hassan", "publish_date": "2025-09-04T00:00:00Z", "category": "Environment"},
    {"title": "Ocean Cleanup Technologies", "content": "Innovative systems for removing plastic debris from oceans are showing promising results in pilot programs.", "author": "Prof. Maria Petrova", "publish_date": "2025-09-05T00:00:00Z", "category": "Environment"},
    
    # Culture & Arts (6 articles)
    {"title": "Immersive Theater Experiences", "content": "Interactive performances are blending theater with virtual reality to create unique audience experiences.", "author": "Valentina Rossi", "publish_date": "2025-09-06T00:00:00Z", "category": "Culture"},
    {"title": "Cultural Heritage Documentation", "content": "Digital archiving and 3D scanning are preserving historical sites and artifacts for future generations.", "author": "James O'Connell", "publish_date": "2025-09-07T00:00:00Z", "category": "Culture"},
    {"title": "Street Food Evolution", "content": "Food trucks and pop-up restaurants are transforming urban dining scenes and culinary innovation.", "author": "Nadia Gomez", "publish_date": "2025-09-08T00:00:00Z", "category": "Culture"},
    {"title": "AI in Creative Writing", "content": "Artificial intelligence tools are assisting authors with idea generation and collaborative storytelling.", "author": "Dr. Victor Liao", "publish_date": "2025-09-09T00:00:00Z", "category": "Culture"},
    {"title": "Festival Technology Integration", "content": "Music and arts festivals are using apps, cashless payments, and AR to enhance attendee experiences.", "author": "Emma Reynolds", "publish_date": "2025-09-10T00:00:00Z", "category": "Culture"},
    {"title": "Artisan Craft Revival", "content": "Traditional handcrafts are experiencing renewed interest through online marketplaces and workshops.", "author": "Valentina Rossi", "publish_date": "2025-09-11T00:00:00Z", "category": "Culture"},
    
    # Economics & Business (7 articles)
    {"title": "B-Corporation Movement Growth", "content": "Benefit corporations are balancing profit with social and environmental impact in business operations.", "author": "Anika Gupta", "publish_date": "2025-09-12T00:00:00Z", "category": "Economics"},
    {"title": "Supply Chain Transparency", "content": "Blockchain and tracking technologies are providing visibility into product origins and ethical sourcing.", "author": "Dr. Luis Fernandez", "publish_date": "2025-09-13T00:00:00Z", "category": "Economics"},
    {"title": "Social Impact Bonds", "content": "Innovative financing mechanisms are funding social programs with measurable outcomes and returns.", "author": "Prof. Claire Dubois", "publish_date": "2025-09-14T00:00:00Z", "category": "Economics"},
    {"title": "Sharing Economy Regulation", "content": "Cities are developing frameworks to manage ride-sharing, home-sharing, and other platform economies.", "author": "Jun Park", "publish_date": "2025-09-15T00:00:00Z", "category": "Economics"},
    {"title": "Green Finance Innovation", "content": "Environmental, social, and governance investing is attracting mainstream investors and institutional funds.", "author": "Haruto Yamamoto", "publish_date": "2025-09-16T00:00:00Z", "category": "Economics"},
    {"title": "Cross-Border E-commerce", "content": "International online trade is growing rapidly with improved logistics and payment processing systems.", "author": "Anika Gupta", "publish_date": "2025-09-17T00:00:00Z", "category": "Economics"},
    {"title": "Economic Recovery Strategies", "content": "Post-pandemic economic policies are focusing on resilience, diversity, and sustainable growth models.", "author": "Dr. Luis Fernandez", "publish_date": "2025-09-18T00:00:00Z", "category": "Economics"},
    
    # Education & Learning (3 articles)
    {"title": "Microlearning Platforms", "content": "Bite-sized educational content is making learning more accessible and fitting into busy schedules.", "author": "Dr. Nina Feldman", "publish_date": "2025-09-19T00:00:00Z", "category": "Education"},
    {"title": "Global Classroom Connections", "content": "Video conferencing is enabling students worldwide to collaborate on projects and share cultural perspectives.", "author": "Haruto Yamamoto", "publish_date": "2025-09-20T00:00:00Z", "category": "Education"},
    {"title": "Skills-Based Assessment Methods", "content": "Alternative evaluation approaches are focusing on practical abilities rather than traditional testing.", "author": "Priya Kumari", "publish_date": "2025-09-21T00:00:00Z", "category": "Education"},
    
    # Sports & Fitness (2 articles)
    {"title": "Mental Training in Athletics", "content": "Sports psychology and mindfulness techniques are becoming integral parts of athletic training programs.", "author": "Dr. Kenji Tanaka", "publish_date": "2025-09-22T00:00:00Z", "category": "Sports"},
    {"title": "Community Sports Programs", "content": "Local initiatives are using sports to build social connections and promote healthy lifestyles for all ages.", "author": "Emma Reynolds", "publish_date": "2025-09-23T00:00:00Z", "category": "Sports"}
    ]

    print("Batch inserting articles...")
    
    # Get the articles collection
    articles_collection = client.collections.get("Article")
    
    # Insert articles in batches
    batch_size = 2  # Small batch size for demonstration
    total_articles = len(articles)
    successful_inserts = 0
    
    for i in range(0, total_articles, batch_size):
        batch = articles[i:i + batch_size]
        objects_to_add = []
        
        for article in batch:
            try:
                # Prepare the article data
                article_data = {
                    "title": article["title"],
                    "content": article["content"],
                    "author": article["author"],
                    "publish_date": article["publish_date"],
                    "category": article["category"]
                }
                objects_to_add.append(article_data)
            except Exception as e:
                print(f"Error preparing article '{article.get('title', 'unknown')}': {str(e)}")
        
        if objects_to_add:
            try:
                # Insert the batch
                result = articles_collection.data.insert_many(objects_to_add)
                successful_inserts += len(result.uuids)
                print(f"Inserted batch of {len(objects_to_add)} articles. Total inserted: {successful_inserts}/{total_articles}")
            except Exception as e:
                print(f"Error inserting batch: {str(e)}")
    
    print(f"\nData insertion complete. Successfully inserted {successful_inserts} out of {total_articles} articles.")
    
    # Verify the count
    try:
        count = articles_collection.aggregate.over_all(total_count=True).total_count
        print(f"\nTotal articles in Weaviate: {count}")
    except Exception as e:
        print(f"Error getting article count: {str(e)}")

    # Example query to verify data
    try:
        print("\nSample query results (searching for 'technology'):")
        result = articles_collection.query.near_text(
            query="technology",
            limit=2,
            return_properties=["title", "author", "category"]
        )
        print_article_info(result.objects)
    except Exception as e:
        print(f"Error running sample query: {str(e)}")

    # Get all articles
    try:
        print("\nAll articles in the collection:")
        all_articles = articles_collection.query.fetch_objects(limit=10)
        print_article_info(all_articles.objects)
    except Exception as e:
        print(f"Error fetching all articles: {str(e)}")

except Exception as e:
    print(f"An error occurred: {str(e)}")
    raise

finally:
    # Always close the client when done
    if 'client' in locals() and client is not None:
        client.close()
        print("\nWeaviate client connection closed.")