import json


def lowercase(s):
    return s[0] + s[1:].lower()


def normalizeData(raw_path, match_json_path, output_path):
    """
    Format:
    Match xxx
    Wind 0..3
    Player 1 Deal XX XX ...
    Player 1 Draw XX
    Player 1 Play XX
    Player 1 Chi XX
    Player 1 Peng/Gang XX Ignore Player 3 Chi
    Player 1 BuGang XX
    Player 1 Hu XX
    Fan 100 XX*2+XX*3+XX*1
    Huang
    Score 0 0 0 0
    """
    output = open(output_path, "w+")
    with open(match_json_path) as f:
        matches = json.load(f)
        matches = matches["ids"]
    matches.sort()
    l = 0
    for match in matches:
        l += 1
        if l % 1024 == 0:
            print("Processing match %d of %d ..." % (l, len(matches)))
        with open("./{}/{}.json".format(raw_path, match)) as f:
            try:
                log = json.load(f)
            except Exception as e:
                print(e)
                print(f)
        if log["logs"][-1]["output"]["display"]["action"] not in ["HU", "HUANG"]:
            continue
        print("Match", match, file=output)
        last = None
        angang = False
        for i in range(0, len(log["logs"]), 2):
            display = log["logs"][i]["output"]["display"]
            if display["action"] == "INIT":
                print("Wind", display["quan"], file=output)
            elif display["action"] == "DEAL":
                for i in range(4):
                    print("Player", i, "Deal", *(display["hand"][i]), file=output)
            elif display["action"] == "DRAW":
                print("Player", display["player"], "Draw", display["tile"], file=output)
                angang = True
            elif display["action"] == "PLAY":
                print("Player", display["player"], "Play", display["tile"], file=output)
                last = display["tile"]
                angang = False
            elif display["action"] == "CHI":
                print(
                    "Player", display["player"], "Chi", display["tileCHI"], file=output
                )
                print("Player", display["player"], "Play", display["tile"], file=output)
                last = display["tile"]
            elif display["action"] == "PENG":
                print("Player", display["player"], "Peng", last, end="", file=output)
                for j in range(4):
                    jaction = log["logs"][i - 1][str(j)]["response"]
                    if j != display["player"] and jaction != "PASS":
                        jaction = (
                            lowercase(jaction.split()[0])
                            + " "
                            + (
                                jaction.split()[1]
                                if jaction.startswith("CHI")
                                else last
                            )
                        )
                        print("", "Ignore", "Player", j, jaction, end="", file=output)
                print(file=output)
                print("Player", display["player"], "Play", display["tile"], file=output)
                last = display["tile"]
            elif display["action"] == "GANG":
                print(
                    "Player",
                    display["player"],
                    "AnGang" if angang else "Gang",
                    display["tile"],
                    end="",
                    file=output,
                )
                for j in range(4):
                    jaction = log["logs"][i - 1][str(j)]["response"]
                    if j != display["player"] and jaction != "PASS":
                        jaction = (
                            lowercase(jaction.split()[0])
                            + " "
                            + (
                                jaction.split()[1]
                                if jaction.startswith("CHI")
                                else last
                            )
                        )
                        print("", "Ignore", "Player", j, jaction, end="", file=output)
                print(file=output)
            elif display["action"] == "BUGANG":
                print(
                    "Player", display["player"], "BuGang", display["tile"], file=output
                )
                last = display["tile"]
            elif display["action"] == "HUANG":
                print("Huang", file=output)
                print("Score", 0, 0, 0, 0, file=output)
            elif display["action"] == "HU":
                print("Player", display["player"], "Hu", last, end="", file=output)
                for j in range(4):
                    jaction = log["logs"][i - 1][str(j)]["response"]
                    if j != display["player"] and jaction != "PASS" and jaction != "":
                        jaction = (
                            lowercase(jaction.split()[0])
                            + " "
                            + (
                                jaction.split()[1]
                                if jaction.startswith("CHI")
                                else last
                            )
                        )
                        print("", "Ignore", "Player", j, jaction, end="", file=output)
                print(file=output)
                print(
                    "Fan",
                    display["fanCnt"],
                    "+".join([f["name"] + "*" + str(f["cnt"]) for f in display["fan"]]),
                    file=output,
                )
                print("Score", *display["score"], file=output)
        print(file=output)


if __name__ == "__main__":
    raw_path = "../data/mahjongNeXt/mahjongNeXt_raw"
    match_json_path = "../data/mahjongNeXt/mahjongNeXt_32768_match_id.json"
    output_path = "../data/mahjongNeXt/mahjongNeXt.txt"
    normalizeData(raw_path, match_json_path, output_path)
