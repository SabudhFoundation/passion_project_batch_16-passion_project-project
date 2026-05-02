# Maps (Domain_Index, Domain_Title, Main_Score_Key, [ (Metric_Label, JSON_Path) ])
domains = [
    (
        "1",
        "Sanitation & Waste Management", 
        "sanitation.odf_declaration_status", # You can set ODF status as the "main" indicator
        [
            ("ODF Declaration Status", "sanitation.odf_declaration_status"),
            ("Community Compost Pits", "sanitation.community_compost_pits"),
            ("Waste Collection Vehicles", "sanitation.vehicles_for_collection_&_transportation_of_waste"),
            ("Waste Collection & Segregation Sheds", "sanitation.waste_collection_and_segregation_sheds_in_the_village"),
            ("Community Segregation Bins", "sanitation.segregation_bins_at_community_places_in_the_village"),
            ("Community Bio-gas Plants", "sanitation.community_bio-gas_plants_-_under_other_schemes"),
            ("Household Compost Pits", "sanitation.compost_pits"),
            ("Household Biogas Plants", "sanitation.biogas_plants"),
            ("Waste Fed to Cattle", "sanitation.feeding_to_cattle"),
            ("Household Bins", "sanitation.household_bins"),
            ("Community Soak/Leach/Magic Pits", "sanitation.community_soak/leach/magic_pits"),
            ("Drainage Facility Available", "sanitation.drainage_facility_available_in_village"),
            ("Community Grey Water Management", "sanitation.community_grey_water_management_systems"),
            ("Leach Pits", "sanitation.leach_pits"),
            ("Magic Pits", "sanitation.magic_pits"),
            ("Soak Pits", "sanitation.soak_pits"),
            ("Kitchen Gardens", "sanitation.kitchen_garden"),
            ("Duckweed Ponds", "sanitation.duckweed_pond")
        ]
    ),
    (
        "2",
        "Governance & Sustainable Development", 
        "governance.overall_score", # The main score
        [
            ("Overall Score", "governance.overall_score"),
            ("Overall Grade", "governance.overall_grade"),
            ("Poverty Free Score", "governance.t1_poverty_free_score"),
            ("Healthy Village Score", "governance.t2_healthy_score"),
            ("Child Friendly Score", "governance.t3_child_friendly_score"),
            ("Water Sufficient Score", "governance.t4_water_sufficient_score"),
            ("Clean & Green Score", "governance.t5_clean_green_score"),
            ("Self Sufficient Score", "governance.t6_self_sufficient_score"),
            ("Socially Just Score", "governance.t7_socially_just_score"),
            ("Good Governance Score", "governance.t8_good_governance_score"),
            ("Women Friendly Score", "governance.t9_women_friendly_score")
        ]
    ),
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
    ]),
]