with section("parse"):
    additional_commands = {
        "rosidl_generate_interfaces": {
            "pargs": "1+",
            "flags": [
                "ADD_LINTER_TESTS",
                "SKIP_INSTALL",
                "SKIP_GROUP_MEMBERSHIP_CHECK",
            ],
            "kwargs": {
                "LIBRARY_NAME": 1,
                "DEPENDENCIES": "1+",
            },
        },
        "rclcpp_components_register_node": {
            "pargs": 1,
            "kwargs": {
                "PLUGIN": 1,
                "EXECUTABLE": 1,
                "RESOURCE_INDEX": 1,
            },
        },
    }

with section("lint"):
    disabled_codes = [
        "C0113",
    ]

with section("format"):
    dangle_parens = True
    line_ending = "unix"
    line_width = 80
    tab_size = 2
    keyword_case = "upper"

with section("markup"):
    bullet_char = "-"
