import {app} from "../../scripts/app.js";

const SPECIFIC_WIDTH = 400; // Set to desired width

function setNodeWidthForMappedTitles(node) {
    node.setSize([SPECIFIC_WIDTH, node.size[1]]);
}


app.registerExtension({
    name: "facechain.appearance",
    async beforeRegisterNodeDef(nodeType, nodeData, _app) {
        if (nodeData.name === "FC StyleLoraLoad") {
            nodeType.prototype.onNodeCreated = function () {
                setNodeWidthForMappedTitles(this)
            };
        }
    },
});