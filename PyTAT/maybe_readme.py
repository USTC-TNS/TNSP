#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2023-2024 Hao Zhang<zh970205@mail.ustc.edu.cn>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#


def dynamic_metadata(field, settings=None):

    if field != "readme":
        msg = "Only the 'readme' field is supported"
        raise ValueError(msg)

    if settings:
        msg = "No inline configuration is supported"
        raise ValueError(msg)

    try:
        with open("README.md", "rt", encoding="utf-8") as file:
            result = file.read()
    except FileNotFoundError:
        result = "empty description"

    return {"content-type": "text/markdown", "text": result}
