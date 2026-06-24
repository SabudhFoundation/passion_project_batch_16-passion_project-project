# --- Localization Dictionary (Speeds up the app significantly) ---
PUNJABI_LABELS = {
    # Proper Nouns & UI Labels (Add any other districts here)
    "Ropar": "ਰੋਪੜ",
    "District": "ਜ਼ਿਲ੍ਹਾ", # Change to "ਜਿਲ੍ਹਾ" or "ਜ਼ਿਲਾ" if you prefer
    "Block": "ਬਲਾਕ",
    "Assessment Date": "ਮੁਲਾਂਕਣ ਤਾਰੀਖ",
    "UDISE Code": "ਯੂ-ਡਾਈਸ ਕੋਡ",
    
    # Sanitation & Waste Management
    "ODF Status": "ਓਡੀਐਫ ਸਥਿਤੀ",
    "Waste Segregation Sheds": "ਕੂੜਾ ਵੱਖ ਕਰਨ ਵਾਲੇ ਸ਼ੈੱਡ",
    
    # Schools
    "Safety and Hygiene": "ਸੁਰੱਖਿਆ ਅਤੇ ਸਫਾਈ",
    "Stimulating School Environment": "ਸਕੂਲ ਦੇ ਵਾਤਾਵਰਣ ਨੂੰ ਉਤੇਜਿਤ ਕਰਨਾ",
    "Physical Development Opportunities": "ਸਰੀਰਕ ਵਿਕਾਸ ਦੇ ਮੌਕੇ",
    "Smart School Facilities": "ਸਮਾਰਟ ਸਕੂਲ ਸਹੂਲਤਾਂ",
    "Domain1 Score Percentage": "ਡੋਮੇਨ 1 ਸਕੋਰ ਪ੍ਰਤੀਸ਼ਤ",
    "Domain2 Score Percentage": "ਡੋਮੇਨ 2 ਸਕੋਰ ਪ੍ਰਤੀਸ਼ਤ",
    "Domain3 Score Percentage": "ਡੋਮੇਨ 3 ਸਕੋਰ ਪ੍ਰਤੀਸ਼ਤ",
    "Domain4 Score Percentage": "ਡੋਮੇਨ 4 ਸਕੋਰ ਪ੍ਰਤੀਸ਼ਤ",
    
    # Villages
    "Sanitation & Waste Management": "ਸੈਨੀਟੇਸ਼ਨ ਅਤੇ ਕੂੜਾ ਪ੍ਰਬੰਧਨ",
    "Governance & LSDG Performance": "ਗਵਰਨੈਂਸ ਅਤੇ LSDG ਪ੍ਰਦਰਸ਼ਨ",
    "Water Security (JJM)": "ਜਲ ਸੁਰੱਖਿਆ (JJM)",
    "Employment (MGNREGA)": "ਰੁਜ਼ਗਾਰ (ਮਨਰੇਗਾ)",
    "ODF Status": "ਓਡੀਐਫ ਸਥਿਤੀ",
    "Waste Segregation Sheds": "ਕੂੜਾ ਵੱਖ ਕਰਨ ਵਾਲੇ ਸ਼ੈੱਡ",
    "Drainage Facility": "ਪਾਣੀ ਨਿਕਾਸੀ ਸਹੂਲਤ",
    "Community Compost Pits": "ਸਾਂਝੇ ਖਾਦ ਟੋਏ",
    "Overall Category": "ਕੁੱਲ ਸ਼੍ਰੇਣੀ",
    "Poverty Free Score": "ਗਰੀਬੀ ਮੁਕਤ ਸਕੋਰ",
    "Healthy Village Score": "ਸਿਹਤਮੰਦ ਪਿੰਡ ਸਕੋਰ",
    "Child Friendly Score": "ਬਾਲ ਅਨੁਕੂਲ ਸਕੋਰ",
    "Socially Just Score": "ਸਮਾਜਿਕ ਨਿਆਂ ਸਕੋਰ",
    "JJM Certification Status": "JJM ਪ੍ਰਮਾਣੀਕਰਨ ਸਥਿਤੀ",
    "PWS Available": "PWS ਉਪਲਬਧ ਹੈ",
    "Total Households": "ਕੁੱਲ ਘਰ",
    "Total Tap Connections": "ਕੁੱਲ ਟੂਟੀ ਕਨੈਕਸ਼ਨ",
    "Total Registered HH": "ਕੁੱਲ ਰਜਿਸਟਰਡ ਘਰ",
    "Total Registered Persons": "ਕੁੱਲ ਰਜਿਸਟਰਡ ਵਿਅਕਤੀ",
    "SC Persons Employed": "ਰੁਜ਼ਗਾਰ ਪ੍ਰਾਪਤ SC ਵਿਅਕਤੀ",
    "Female Persons Employed": "ਰੁਜ਼ਗਾਰ ਪ੍ਰਾਪਤ ਔਰਤਾਂ"
}

# ==========================================
# VILLAGE REPORT CONSTANTS
# ==========================================

# Maps (Domain_Index, Domain_Title, Main_Score_Key, [ (Metric_Label, JSON_Path) ])
VILLAGE_DOMAINS = [
    (1, "Sanitation & Waste Management", "sanitation.odf_declaration_status", [
        ("ODF Status", "sanitation.odf_declaration_status"),
        ("Waste Segregation Sheds", "sanitation.waste_collection_and_segregation_sheds_in_the_village"),
        ("Drainage Facility", "sanitation.drainage_facility_available_in_village"),
        ("Community Compost Pits", "sanitation.community_compost_pits")
    ]),
    (2, "Governance & LSDG Performance", "governance.overall_score", [
        ("Overall Category", "governance.overall_category"),
        ("Poverty Free Score", "governance.t1_poverty_free_score"),
        ("Healthy Village Score", "governance.t2_healthy_score"),
        ("Child Friendly Score", "governance.t3_child_friendly_score"),
        ("Socially Just Score", "governance.t7_socially_just_score")
    ]),
    (3, "Water Security (JJM)", "water_security.jjm_status", [
        ("JJM Certification Status", "water_security.jjm_status"),
        ("PWS Available", "water_security.is_pws_available"),
        ("Total Households", "water_security.total_households"),
        ("Total Tap Connections", "water_security.total_tap_connections")
    ]),
    (4, "Employment (MGNREGA)", "employment.total_registered_hh", [
        ("Total Registered HH", "employment.total_registered_hh"),
        ("Total Registered Persons", "employment.total_registered_persons"),
        ("SC Persons Employed", "employment.sc_persons"),
        ("Female Persons Employed", "employment.female_persons")
    ])
]


# ==========================================
# SCHOOL REPORT CONSTANTS
# ==========================================

# Kept named 'domains' so it does not break your existing utils.py logic
domains = [
    (1, "Safety and Hygiene", "Domain1 Score", "Domain1 Score Percentage", [
        ("Toilets", "Infra_Toilets"), 
        ("Handwash Facilities", "Infra_Handwash"), 
        ("Drinking Water", "Infra_DrinkingWater"), 
        ("Mid day meal", "Infra_MiddayMeal"), 
        ("School Building", "Infra_SchoolBuilding"),
        ("Safe Surrounding", "Infra_SafeSurroundings")
    ]),
    (2, "Stimulating School Environment", "Domain2 Score", "Domain2 Score Percentage", [
        ("Classroom Resources", "Env_ClassroomResources"), 
        ("Wall Painting", "Env_WallPainting"), 
        ("Print Rich Classrooms", "Env_PrintRich"), 
        ("Green Premises", "Env_GreenPremises")
    ]), 
    (3, "Physical Development Opportunities", "Domain3 Score", "Domain3 Score Percentage", [
        ("Playground", "Physical_Playground"), 
        ("Sports Equipment", "Physical_SportsEquipment"), 
        ("Other Physical Activity Spaces", "Physical_OtherSpaces")
    ]), 
    (4, "Smart School Facilities", "Domain4 Score", "Domain4 Score Percentage", [
        ("Library", "Smart_Library"), 
        ("Digital Learning Resources", "Smart_DigitalResources"), 
        ("Education Park", "Smart_EducationPark"),
        ("Centre Resources", "TRC_Resources")
    ]),
]

mapping_dict_toilets = {
    'Unavailable or Unusable': 1,
    'Separate toilets for boys & girls, but insufficient and not very clean': 2,
    'Separate toilets for boys & girls, but insufficient and poorly maintained':2,
    'Sufficient toilets, (40:1) for boys and (25:1) for girls and are well maintained': 3,
    'Sufficient toilets, (40:1) for boys and (25:1) for girls and are clean and repaired as per need.':3,
    'Sufficient toilets, (40:1) for boys, (25:1) for girls and 1 for disabled children and have regular water supply and regularly cleaned': 4
}

mapping_dict_handwash = {
    'Insufficient water supply and inadequate hand-wash stations; no soap': 1,
    'Insufficient water supply and inadequate hand-wash stations':1,
    'Sufficient hand-wash stations (40:1) with regular water supply but no/insufficient soap': 2,
    'Sufficient supply of water & soap, stations cleaned regularly, and teachers tell importance of hand-washing': 3,
    "Hand-wash stations cleaned daily, school leaders/teachers monitor and actively promote students' personal hygiene through a variety of initiatives": 4
}

mapping_dict_Water = {
    'Unavailable or insufficient or unfit for drinking': 1,
    'Drinking water is not available':1,
    'Unreliable supply of drinking water that may not have been quality checked': 2,
    'Unreliable supply of drinking water and has not been checked for quality':2,
    'Regular supply of unpurified water; cleanliness around drinking facilities': 3,
    'Regular supply of purified water; equipment & premises well maintained': 4
}

mapping_dict_meal = {
    'Kitchen is unavailable or poorly maintained': 1,
    'Kitchen is well maintained but different sections for cooking, storage, washing not available': 2,
    'Kitchen has different sections to cook, store, wash but staff is insufficient, or same food items everyday': 3,
    'MDM staff is sufficient, and a variety of food items are served': 4
}

mapping_dict_schoolBuilding = {
    'Major cracks, leakages in one or more functional rooms, needs urgent attention': 1,
    'Small cracks, minor leakages in one or more of functional rooms': 2,
    'No cracks, leakage in all functional rooms & Panchayat gets building checked regularly': 3,
    'No cracks, leakage in any room, and school has PWD safety approval': 4
}

mapping_dict_safeSurrounding = {
    'Anyone can enter or children roam outside the school as no boundary wall or gate': 1,
    'Boundary wall and Gate are present but needs repair': 2,
    'Children secure within school; boundary wall, gate well maintained by school staff': 3,
    "Children's security: prime concern of community; they regularly repair the boundary wall and gate": 4
}

mapping_dict_ClassRoom = {
    'Classrooms: crowded, poorly ventilated. Furniture is insufficient. No dedicated subject corner and insufficient teaching-learning material': 1,
    'Children have space to move around in their classroom, but ventilation is improper; furniture is adequate but poorly maintained. Dedicated subject corners are created but teaching-learning material is out-dated.':2,
    'Children have space to move around in their classrooms but ventilation is improper; furniture is adequate but poorly maintained. Dedicated subject corners are created but teaching-learning material is out-dated': 2,
    'Majority of classrooms have good ventilation, natural light and fans (where needed); furniture is adequate and well maintained. Teachers have sufficient teaching-learning material which is occasionally used': 3,
    'Majority of classrooms have good ventilation, natural light and fans (where needed); furniture is adequate and well maintained. Teachers have sufficient teaching-learning material available, updated as per latest curiculum.':3,
    'Every classroom has good ventilation, light; furniture is sufficient & age-appropriate. Teaching-learning material is available as per latest curriculum.': 4,
    'Every classroom has good ventilation, light; furniture is sufficient & age-appropriate. Teaching-learning material is regularly used.':4
}

mapping_dict_Wall = {
    'All the walls of the school either not painted or painted with dull colors': 1,
    'All the walls are painted but ignore the BaLa guidelines completely': 2,
    "All the walls are painted but didn't use BaLA guidelines at all":2,
    "All the walls of the school are well maintained but didn't follow BaLA guidelines completely": 3,
    'All the walls of the school are painted according to BaLA guidelines': 4
}

mapping_dict_print = {
    'No print material visible in the classrooms': 1,
    'Some print material is visible with no designated spaces to encourage interaction and/or learning': 2,
    'Some print material is visible but poorly placed, with no designated spaces to encourage interaction and/or learning':2,
    "Classroom is print rich with designated space but children's active involvement is minimal or absent": 3,
    "Classroom is print rich, with materials arranged in a sequence, with designated space but children's active involvement is minimal or absent":3,
    'Classroom is print rich with designated spaces and active involvement of children': 4,
    'Classroom is print rich with designated spaces and children seems to actively engage with the material':4
}

mapping_dict_plant = {
    'No plantation within the school premises': 1,
    "Some signs of plantation but it's poorly maintained": 2,
    'Planned plantation is visible and well maintained by school staff': 3,
    'Planned plantation is visible and well maintained by community members': 4
}

mapping_dict_playgroud = {
    'No access to playground': 1,
    'Playground of inadequate size is available': 2,
    'Playground of adequate size available but minimal student participation': 3,
    'Students participate in a variety of games/ sports in a planned manner': 4
}

mapping_dict_sports = {
    'No access to sports equipment': 1,
    'Adequate material and equipment available only for a few games': 2,
    'Adequate sports equipment and material available for a variety of games': 3,
    'Facility for training/coaching for sports is available': 4
}

mapping_dict_Otherspace = {
    'No physically stimulating spaces available': 1,
    'Open area other than playground available for physical activities but not accessible to children': 2,
    'Open area, other than playground, accessible to children but needs maintenance': 3,
    'Open area, other than playground, well maintained and accessed by children regularly': 4
}

mapping_dict_Library = {
    'No books available': 1,
    'Some books are available but with no designated space for the library': 2,
    'Library is available with a designated space but insufficient number of books, is well-managed, and used regularly during library classes': 3,
    'Teachers and school leaders actively promote the use of a well-resourced, and well-managed library': 4
}

mapping_dict_digital = {
    'Computers and digital learning resources are not available for students': 1,
    'Insufficient number of computers with some software and digital learning resources are available and occassionally used by students as per syllabus requirements': 2,
    'Sufficient number of computers with software and digital learning resources are regularly used by students in allocated periods': 3,
    'Every student gets an opportunity to use the computer as well as digital learning resources on their own in addition to teacher-led engagements': 4
}

mapping_dict_Park = {
    'No education park available': 1,
    'Educational park is available but not used': 2,
    'Education park is used by children but needs repair': 3,
    'Education park is functional and used by children regularly': 4
}

mapping_dict_trc = {
    'There is no resource center': 1,
    "Not a CHT school":1,
    'A common room for teachers is available but with inadequate resources': 2,
    'Common room, equipped with basic resources, available to teachers': 3,
    'Common room, equipped with a variety of resources, available to teachers': 4
}

# ---------- FEATURE COLUMNS (explicit) ----------
safety_cols = [
    "Safety and Hygiene / Toilets",
    "Safety and Hygiene / Handwash Facilities",
    "Safety and Hygiene / Drinking Water",
    "Safety and Hygiene / Mid day meal",
    "Safety and Hygiene / School Building",
    "Safety and Hygiene / Safe Surrounding"
]

stim_cols = [
    "Stimulating School Environment/ Classroom Resources",
    "Stimulating School Environment/ Wall Painting",
    "Stimulating School Environment/ Print Rich Classrooms",
    "Stimulating School Environment/ Green Premises"
]

physical_cols = [
    "Physical Development Opportunities/ Playground",
    "Physical Development Opportunities/ Sports Equipment",
    "Physical Development Opportunities/ Other Physical Activity Spaces"
]

smart_cols = [
    "Smart School Facilities/Library",
    "Smart School Facilities/ Digital Learning Resources",
    "Smart School Facilities/Education Park",
    "Smart School Facilities/Centre Resources"
]

object_to_int_col = ['Students_PrePrimary_Girls', 'Students_PrePrimary_Boys',
    'Students_Grade1_Girls', 'Students_Grade1_Boys',
    'Students_Grade2_Girls', 'Students_Grade2_Boys',
    'Students_Grade3_Girls', 'Students_Grade3_Boys',
    'Students_Grade4_Girls', 'Students_Grade4_Boys',
    'Students_Grade5_Girls', 'Students_Grade5_Boys',
    'Students_Disability_Count','Teacher_Positions_Sanctioned', 
    'Teachers_Present', "Teachers_Deputation", "Teachers_New_Recruits",
    "Children_PrivateSchool", "Children_Anganwadi_0_3","Students_Total_GPS"
]

School_info_cols = [
"School_Name", "UDISE_Code", "Assessment_Date",
"Teacher_Positions_Sanctioned", "Teachers_Present", "Students_Disability_Count",
"Teachers_Deputation", "Teachers_New_Recruits", "Children_PrivateSchool",
"Children_Anganwadi_0_3", "Students_Total_GPS"
]