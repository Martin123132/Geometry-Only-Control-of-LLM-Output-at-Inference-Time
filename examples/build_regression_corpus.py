"""Build the lightweight public MBT-5 offline regression corpus."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


OUTPUT_PATH = Path(__file__).with_name("regression_corpus.jsonl")


def emit_case(case_id: str, references: list[str], bad: str, good: str) -> dict:
    return {
        "id": case_id,
        "references": references,
        "candidates": [bad, good],
        "expected_action": "emit",
        "expected_emitted_text": good,
        "expected_candidate_safe": [False, True],
    }


def block_case(case_id: str, references: list[str], bad_candidates: Iterable[str]) -> dict:
    candidates = list(bad_candidates)
    return {
        "id": case_id,
        "references": references,
        "candidates": candidates,
        "expected_action": "block",
        "expected_emitted_text": None,
        "expected_candidate_safe": [False] * len(candidates),
    }


def safe_case(case_id: str, references: list[str], safe_candidates: Iterable[str]) -> dict:
    candidates = list(safe_candidates)
    return {
        "id": case_id,
        "references": references,
        "candidates": candidates,
        "expected_action": "emit",
        "expected_emitted_text": candidates[0],
        "expected_candidate_safe": [True] * len(candidates),
    }


def build_cases() -> list[dict]:
    cases: list[dict] = []

    capitals = [
        ("france", "France", "Paris", "London"),
        ("italy", "Italy", "Rome", "Milan"),
        ("germany", "Germany", "Berlin", "Munich"),
        ("spain", "Spain", "Madrid", "Barcelona"),
        ("japan", "Japan", "Tokyo", "Kyoto"),
        ("canada", "Canada", "Ottawa", "Toronto"),
        ("brazil", "Brazil", "Brasilia", "Rio"),
        ("australia", "Australia", "Canberra", "Sydney"),
        ("egypt", "Egypt", "Cairo", "Alexandria"),
        ("portugal", "Portugal", "Lisbon", "Porto"),
        ("norway", "Norway", "Oslo", "Bergen"),
        ("sweden", "Sweden", "Stockholm", "Gothenburg"),
        ("finland", "Finland", "Helsinki", "Tampere"),
        ("ireland", "Ireland", "Dublin", "Cork"),
        ("poland", "Poland", "Warsaw", "Krakow"),
        ("greece", "Greece", "Athens", "Thessaloniki"),
        ("turkey", "Turkey", "Ankara", "Istanbul"),
        ("mexico", "Mexico", "Mexico City", "Guadalajara"),
        ("argentina", "Argentina", "Buenos Aires", "Cordoba"),
        ("chile", "Chile", "Santiago", "Valparaiso"),
        ("peru", "Peru", "Lima", "Cusco"),
        ("kenya", "Kenya", "Nairobi", "Mombasa"),
        ("morocco", "Morocco", "Rabat", "Casablanca"),
        ("india", "India", "New Delhi", "Mumbai"),
        ("china", "China", "Beijing", "Shanghai"),
    ]
    for key, country, city, wrong_city in capitals:
        cases.append(
            emit_case(
                f"capital_entity_swap_{key}",
                [f"The capital of {country} is {city}."],
                f"The capital of {country} is {wrong_city}.",
                f"The capital of {country} is {city}.",
            )
        )

    for key, country, city, wrong_city in capitals[:5]:
        cases.append(
            block_case(
                f"capital_all_bad_{key}",
                [f"The capital of {country} is {city}."],
                [
                    f"The capital of {country} is {wrong_city}.",
                    f"{wrong_city} is the capital city of {country}.",
                ],
            )
        )

    for key, country, city, _wrong_city in capitals[:5]:
        cases.append(
            safe_case(
                f"capital_supported_paraphrase_{key}",
                [f"The capital of {country} is {city}."],
                [f"{city} is the capital city of {country}."],
            )
        )

    for key, country, city, _wrong_city in capitals[5:]:
        cases.append(
            safe_case(
                f"capital_reverse_paraphrase_{key}",
                [f"{city} is the capital city of {country}."],
                [f"The capital of {country} is {city}."],
            )
        )

    numeric_cases = [
        (
            "water_boiling_point",
            "Water boils at 100 degrees Celsius at sea level.",
            "Water boils at 90 degrees Celsius at sea level.",
        ),
        ("mars_moons", "Mars has two moons.", "Mars has three moons."),
        ("earth_moon", "Earth has one moon.", "Earth has two moons."),
        ("treaty_year", "The Treaty was signed in 1919.", "The Treaty was signed in 1918."),
        (
            "constitution_year",
            "The Constitution was adopted in 1788.",
            "The Constitution was adopted in 1787.",
        ),
        ("landing_year", "Eagle landed on the moon in 1969.", "Eagle landed on the moon in 1968."),
        ("bridge_length", "The bridge is 20 meters long.", "The bridge is 21 meters long."),
        ("sample_weight", "The sample weighs 5 kg.", "The sample weighs 6 kg."),
        ("test_duration", "The test lasts 30 minutes.", "The test lasts 25 minutes."),
        ("battery_voltage", "The battery is 12 volts.", "The battery is 24 volts."),
        ("room_chairs", "The room holds four chairs.", "The room holds five chairs."),
        ("pipe_width", "The pipe is 4 cm wide.", "The pipe is 5 cm wide."),
        ("storage_size", "The archive is 8 g.", "The archive is 9 g."),
        ("sample_count", "The trial used ten samples.", "The trial used nine samples."),
        ("lab_temperature", "The lab temperature is 22 degrees Celsius.", "The lab temperature is 23 degrees Celsius."),
        ("orbit_period", "The orbit period is 12 years.", "The orbit period is 11 years."),
        ("panel_count", "The array uses six panels.", "The array uses seven panels."),
        ("valve_pressure", "The valve pressure is 3 kg.", "The valve pressure is 4 kg."),
        ("run_hours", "The run lasts 2 hours.", "The run lasts 3 hours."),
        ("survey_year", "The survey occurred in 2024.", "The survey occurred in 2025."),
        ("signal_delay", "The signal delay is 9 seconds.", "The signal delay is 8 seconds."),
        ("dataset_rows", "The dataset has 100 rows.", "The dataset has 200 rows."),
        ("batch_size", "The batch uses 32 samples.", "The batch uses 64 samples."),
        ("voltage_reading", "The reading is 5 volts.", "The reading is 6 volts."),
    ]
    for key, reference, bad in numeric_cases:
        cases.append(emit_case(f"numeric_drift_{key}", [reference], bad, reference))

    unit_cases = [
        ("length_miles", "The sample is 5 meters long.", "The sample is 5 miles long."),
        ("temperature_kelvin", "The chamber is 20 degrees Celsius.", "The chamber is 20 degrees Kelvin."),
        ("speed_km", "The car travels at 30 mph.", "The car travels at 30 km."),
        ("mass_grams", "The package weighs 10 kg.", "The package weighs 10 g."),
        ("distance_meters", "The trail is 2 miles long.", "The trail is 2 meters long."),
        ("duration_seconds", "The pulse lasts 3 seconds.", "The pulse lasts 3 hours."),
        ("route_km", "The route is 7 km long.", "The route is 7 miles long."),
        ("height_cm", "The column is 40 cm tall.", "The column is 40 mm tall."),
        ("speed_meters_second", "The flow is 6 m/s.", "The flow is 6 mph."),
        ("temperature_fahrenheit", "The oven is 350 degrees Fahrenheit.", "The oven is 350 degrees Celsius."),
        ("range_kilometers", "The range is 12 kilometers.", "The range is 12 miles."),
        ("cooldown_minutes", "The cooldown is 15 minutes.", "The cooldown is 15 seconds."),
        ("storage_percent", "The storage is 80 percent full.", "The storage is 80 kg full."),
        ("probe_mm", "The probe is 9 mm wide.", "The probe is 9 cm wide."),
        ("run_years", "The mission lasts 5 years.", "The mission lasts 5 hours."),
        ("water_fahrenheit", "Water boils at 212 degrees Fahrenheit.", "Water boils at 212 degrees Celsius."),
    ]
    for key, reference, bad in unit_cases:
        cases.append(emit_case(f"unit_drift_{key}", [reference], bad, reference))

    copular_cases = [
        ("sun_star", "The Sun is a star.", "The Sun is a planet."),
        ("dog_mammal", "A dog is a mammal.", "A dog is a bird."),
        ("whale_mammal", "A whale is a mammal.", "A whale is a fish."),
        ("copper_metal", "Copper is a metal.", "Copper is a gas."),
        ("saturn_planet", "Saturn is a planet.", "Saturn is a star."),
        ("rose_flower", "A rose is a flower.", "A rose is a tree."),
        ("oak_tree", "An oak is a tree.", "An oak is an animal."),
        ("triangle_shape", "A triangle is a shape.", "A triangle is a number."),
        ("ice_solid", "Ice is a solid.", "Ice is a gas."),
        ("oxygen_gas", "Oxygen is a gas.", "Oxygen is a metal."),
        ("gold_metal", "Gold is a metal.", "Gold is a liquid."),
        ("salmon_fish", "A salmon is a fish.", "A salmon is a bird."),
        ("piano_instrument", "A piano is an instrument.", "A piano is a planet."),
        ("square_shape", "A square is a shape.", "A square is a mammal."),
        ("helium_gas", "Helium is a gas.", "Helium is a metal."),
        ("granite_rock", "Granite is a rock.", "Granite is a plant."),
        ("carrot_vegetable", "A carrot is a vegetable.", "A carrot is a mineral."),
        ("sparrow_bird", "A sparrow is a bird.", "A sparrow is a fish."),
        ("lava_molten_rock", "Lava is molten rock.", "Lava is ice."),
        ("bee_insect", "A bee is an insect.", "A bee is a mammal."),
        ("venus_planet", "Venus is a planet.", "Venus is a star."),
        ("ruby_mineral", "A ruby is a mineral.", "A ruby is a mammal."),
        ("oak_plant", "An oak is a plant.", "An oak is a fish."),
        ("shark_fish", "A shark is a fish.", "A shark is a tree."),
        ("eagle_bird", "An eagle is a bird.", "An eagle is a mineral."),
        ("silver_metal", "Silver is a metal.", "Silver is a gas."),
        ("snow_solid", "Snow is a solid.", "Snow is a liquid."),
        ("steam_gas", "Steam is a gas.", "Steam is a metal."),
        ("violin_instrument", "A violin is an instrument.", "A violin is a planet."),
        ("hexagon_shape", "A hexagon is a shape.", "A hexagon is a bird."),
        ("maple_tree", "A maple is a tree.", "A maple is an animal."),
        ("neon_gas", "Neon is a gas.", "Neon is a rock."),
        ("basalt_rock", "Basalt is a rock.", "Basalt is an insect."),
        ("lettuce_vegetable", "Lettuce is a vegetable.", "Lettuce is a mineral."),
        ("trout_fish", "A trout is a fish.", "A trout is a flower."),
    ]
    for key, reference, bad in copular_cases:
        cases.append(emit_case(f"copular_relation_{key}", [reference], bad, reference))

    relation_cases = [
        ("earth_orbits_sun", "Earth orbits the Sun.", "The Sun orbits Earth."),
        ("moon_orbits_earth", "The Moon orbits Earth.", "Earth orbits the Moon."),
        ("dna_contains_genes", "DNA contains genes.", "Genes contain DNA."),
        ("cells_contain_nuclei", "Cells contain nuclei.", "Nuclei contain cells."),
        ("friction_produces_heat", "Friction produces heat.", "Heat produces friction."),
        ("plants_use_sunlight", "Plants use sunlight.", "Sunlight uses plants."),
        ("mammals_need_oxygen", "Mammals need oxygen.", "Oxygen needs mammals."),
        ("exercise_improves_health", "Exercise improves health.", "Health improves exercise."),
        (
            "photosynthesis_releases_oxygen",
            "Photosynthesis releases oxygen.",
            "Photosynthesis releases carbon dioxide.",
        ),
        ("engines_produce_motion", "Engines produce motion.", "Motion produces engines."),
        ("rain_produces_runoff", "Rain produces runoff.", "Runoff produces rain."),
        ("rivers_contain_water", "Rivers contain water.", "Water contains rivers."),
        ("libraries_contain_books", "Libraries contain books.", "Books contain libraries."),
        ("boilers_produce_steam", "Boilers produce steam.", "Steam produces boilers."),
        ("students_use_textbooks", "Students use textbooks.", "Textbooks use students."),
        ("birds_need_oxygen", "Birds need oxygen.", "Oxygen needs birds."),
        ("practice_improves_skill", "Practice improves skill.", "Skill improves practice."),
        ("servers_store_data", "Servers store data.", "Data stores servers."),
        ("fire_produces_smoke", "Fire produces smoke.", "Smoke produces fire."),
        ("batteries_store_energy", "Batteries store energy.", "Energy stores batteries."),
        ("roots_need_water", "Roots need water.", "Water needs roots."),
        ("computers_use_electricity", "Computers use electricity.", "Electricity uses computers."),
        ("clouds_produce_rain", "Clouds produce rain.", "Rain produces clouds."),
        ("authors_use_words", "Authors use words.", "Words use authors."),
        ("hospitals_need_power", "Hospitals need power.", "Power needs hospitals."),
        ("muscles_store_energy", "Muscles store energy.", "Energy stores muscles."),
        ("brakes_produce_friction", "Brakes produce friction.", "Friction produces brakes."),
        ("archives_contain_records", "Archives contain records.", "Records contain archives."),
        ("teachers_use_lessons", "Teachers use lessons.", "Lessons use teachers."),
        ("engines_need_oil", "Engines need oil.", "Oil needs engines."),
        ("practice_improves_memory", "Practice improves memory.", "Memory improves practice."),
        ("plants_store_sugar", "Plants store sugar.", "Sugar stores plants."),
        ("lamps_use_electricity", "Lamps use electricity.", "Electricity uses lamps."),
        ("maps_contain_symbols", "Maps contain symbols.", "Symbols contain maps."),
        ("volcanoes_produce_lava", "Volcanoes produce lava.", "Lava produces volcanoes."),
        ("students_need_sleep", "Students need sleep.", "Sleep needs students."),
        ("software_uses_data", "Software uses data.", "Data uses software."),
    ]
    for key, reference, bad in relation_cases:
        cases.append(emit_case(f"noncopular_relation_{key}", [reference], bad, reference))

    coordinated_cases = [
        (
            "photosynthesis_release_store",
            ["Photosynthesis releases oxygen.", "Photosynthesis stores light energy."],
            "Photosynthesis converts oxygen into carbon dioxide and water.",
            "Photosynthesis releases oxygen and stores light energy.",
        ),
        (
            "plants_use_need",
            ["Plants use sunlight.", "Plants need water."],
            "Plants use oxygen and need sunlight.",
            "Plants use sunlight and need water.",
        ),
        (
            "engines_produce_use",
            ["Engines produce motion.", "Engines use fuel."],
            "Engines produce fuel and use motion.",
            "Engines produce motion and use fuel.",
        ),
    ]
    for key, references, bad, good in coordinated_cases:
        cases.append(emit_case(f"coordinated_relation_{key}", references, bad, good))

    negation_cases = [
        ("water_liquid", "Water is liquid at room temperature.", "Water is not liquid at room temperature."),
        ("sound_medium", "Sound needs a material medium to travel.", "Sound does not need a material medium to travel."),
        ("plants_sunlight", "Plants use sunlight.", "Plants do not use sunlight."),
        (
            "science_measurements",
            "Scientific descriptions use measurements.",
            "Scientific descriptions do not use measurements.",
        ),
        ("light_radiation", "Light is electromagnetic radiation.", "Light is not electromagnetic radiation."),
        ("mammals_oxygen", "Mammals need oxygen.", "Mammals do not need oxygen."),
        ("dna_genes", "DNA contains genes.", "DNA does not contain genes."),
        ("earth_sun", "Earth orbits the Sun.", "Earth does not orbit the Sun."),
        ("copper_metal", "Copper is a metal.", "Copper is not a metal."),
        ("treaty_signed", "The treaty was signed in 1919.", "The treaty was not signed in 1919."),
        ("birds_oxygen", "Birds need oxygen.", "Birds do not need oxygen."),
        ("libraries_books", "Libraries contain books.", "Libraries do not contain books."),
        ("gold_metal", "Gold is a metal.", "Gold is not a metal."),
        ("computers_electricity", "Computers use electricity.", "Computers do not use electricity."),
        ("moon_earth", "The Moon orbits Earth.", "The Moon does not orbit Earth."),
        ("libraries_records", "Archives contain records.", "Archives do not contain records."),
        ("teachers_lessons", "Teachers use lessons.", "Teachers do not use lessons."),
        ("engines_oil", "Engines need oil.", "Engines do not need oil."),
        ("lamps_electricity", "Lamps use electricity.", "Lamps do not use electricity."),
        ("volcanoes_lava", "Volcanoes produce lava.", "Volcanoes do not produce lava."),
        ("students_sleep", "Students need sleep.", "Students do not need sleep."),
        ("venus_planet", "Venus is a planet.", "Venus is not a planet."),
        ("silver_metal", "Silver is a metal.", "Silver is not a metal."),
        ("steam_gas", "Steam is a gas.", "Steam is not a gas."),
        ("maps_symbols", "Maps contain symbols.", "Maps do not contain symbols."),
    ]
    for key, reference, bad in negation_cases:
        cases.append(block_case(f"unsupported_negation_{key}", [reference], [bad]))

    overclaim_cases = [
        ("gravity_solved", "Gravity is described by models.", "Gravity is fully solved by modern physics."),
        ("final_truth", "This is a provisional result.", "This is the final truth."),
        (
            "complete_verified",
            "The experiment has preliminary support.",
            "The experiment is complete and experimentally verified.",
        ),
        ("automatic_final", "The answer is a candidate result.", "The answer is automatically the final result."),
        (
            "gravity_no_connection",
            "Gravity connects mass and energy in the supplied description.",
            "General relativity proves gravity has no connection to mass or energy.",
        ),
    ]
    for key, reference, bad in overclaim_cases:
        cases.append(block_case(f"overclaim_{key}", [reference], [bad]))

    exact_safe = [
        "Earth orbits the Sun.",
        "DNA contains genes.",
        "Water is liquid at room temperature.",
        "The capital of France is Paris.",
        "A dog is a mammal.",
        "Plants use sunlight.",
        "Mammals need oxygen.",
        "Copper is a metal.",
        "The Treaty was signed in 1919.",
        "The sample is 5 meters long.",
        "Venus is a planet.",
        "Archives contain records.",
        "Lamps use electricity.",
        "Volcanoes produce lava.",
        "Silver is a metal.",
        "The capital of India is New Delhi.",
        "The capital of China is Beijing.",
        "Steam is a gas.",
        "Maps contain symbols.",
        "Students need sleep.",
    ]
    for index, reference in enumerate(exact_safe, start=1):
        cases.append(safe_case(f"exact_reference_member_{index:02d}", [reference], [reference]))

    return cases


def main() -> None:
    cases = build_cases()
    with OUTPUT_PATH.open("w", encoding="utf-8", newline="\n") as handle:
        for case in cases:
            handle.write(json.dumps(case, ensure_ascii=True, separators=(",", ":")))
            handle.write("\n")
    print(f"wrote {len(cases)} cases to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
