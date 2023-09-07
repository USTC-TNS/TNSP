#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2022-2023 Hao Zhang<zh970205@mail.ustc.edu.cn>
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

from subprocess import check_output


def get_version():
    version = check_output(["git", "describe"]).decode("utf-8")
    version = version.replace("\n", "").replace("v", "").replace("-", ".post", 1).replace("-", "+")
    return version


def main():
    version = get_version()
    with open("CHANGELOG.md", "rt", encoding="utf-8") as file:
        changelog = file.read()
    maybe = [log for log in changelog.split("\n## ") if f"...v{version}" in log]
    with open("release_description.md", "wt", encoding="utf-8") as file:
        for log in maybe:
            print("\n".join(log.split("\n")[1:]), file=file)


if __name__ == "__main__":
    main()
