//The model was trained by David Riebschläger's algorithm in an extern C# project with Keras.NET. 
//This JavaScript code and the whole page is also written by David Riebschläger. 
"use strict";
const boardWide = 10;
const boardHeight = 8;
const numberOfFields = boardWide * boardHeight;
const board = new Array(numberOfFields);
const gapInformation = new Array(boardWide); //specifies the next field index for the selected column
const animationBoard = new Array(numberOfFields + boardWide);
const moveHistory = new Array(numberOfFields + 1);

const bottom = boardWide;
const left = -1;
const right = 1;
const bottomLeft = bottom + left;
const bottomRight = bottom + right;

const p1MoveText = "Yellow's turn!";
const p2MoveText = "Cyan's turn!";

const winDistance = 4;

const empty = 0;
const p1Token = 1;
const p2Token = 2;
const highlightArrow = 3;
const arrowRowOffset = 0.15;
const activationLabelRowOffset = 0.08;
const arrowSizeFactor = 0.7;
const tokenRowOffset = 0.01;
const p1Color = "rgb(255, 255, 0)";
const p2Color = "rgb(0, 255, 255)";

const refreshBoardFrequence = 30; //corresponds to 30 to 34FPS, the actual refresh only takes place when there is a real change, at points that have really changed, this is only the check frequency.
const animationTimeFrequence = 90;
const animationTimeArrowOffset = 180;
const gameEndFrequence = 300;
const delayToNextMove = 260;
const nKeyCountResetDelayTime = 2000;
const aiDelayTime = 720;
const initFinishTimeMoveAcceptanceDelay = 50;
var initFinishTime;

//originally I wrote comments in german, so some comments are still german here...
var nKeyCount; //wird genutzt um die "neues spiel" tastatur timings zu steuern
var lastMoveExecutionTime;

var date;
var inAninmationMode; //es werden keine nachrichten angezeigt, solange die animation läuft
var p1InAnimationMode;
var p2InAnimationMode;
var p1NextAnimationPosition;
var p2NextAnimationPosition;
var p1LastAnimationTime;
var p2LastAnimationTime;

var boardPositionX;
var boardPositionY;
var singleFieldHeight;
var singleFieldWidth;
var arrowWide;
var arrowHeight;

var currentPlayer;
var isAIByPlayerId = [false, true];
var currentPlayerText;
var lastMoveGap;
var gameIsOver;
var winMassageShown; //der sieg wird nur einmal angezeigt
var moveNumber;

var kerasAIModel;
var savedAIMove;
var stopAI = false;
var prediction;

window.onload = function() {
    this.initGame();
    document.getElementById("newGameButton").addEventListener("click", initGame);
    document.getElementById("board").addEventListener("click", onMouseClick);
    document.getElementById("comP1Box").addEventListener("change", onComBoxChange);
    document.getElementById("comP2Box").addEventListener("change", onComBoxChange);
    window.addEventListener("resize", refreshViewTwice);
}

async function initGame() {
    if (savedAIMove > -1) {
        stopAI = true;
        savedAIMove = -1;
        inAninmationMode = false;
        clearTimeouts();
    }
    stopAI = false;
    prediction = null;
    currentPlayer = 1;
    currentPlayerText = "model is loading...";
    gameIsOver = false;
    inAninmationMode = false;
    p1InAnimationMode = false;
    p2InAnimationMode = false;
    moveNumber = 0;
    lastMoveGap = -1;
    lastMoveExecutionTime = 0;
    resetNKeyCount();
    initBoard();
    initCheckBoxValues();
    refreshTextView();
    if (kerasAIModel === undefined) {
        document.getElementById("loadingMessage").style.visibility = "visible";
        kerasAIModel = await loadKerasModel();
        predictMoveEvaluations();
    }
    initBoardView();
    initMoveHistory();
    refreshView();
    currentPlayerText = p1MoveText;
    refreshTextView();
    updateAi();
    initFinishTime = new Date().getTime();

}

function clearTimeouts() {
    let highestTimeoutId = setTimeout(clearTimeouts, 1000);
    disableAnimationMode();
    for (let i = 0; i <= highestTimeoutId; i++) {
        clearTimeout(i);
    }
}

function onComBoxChange() {
    isAIByPlayerId[0] = document.getElementById("comP1Box").checked;
    isAIByPlayerId[1] = document.getElementById("comP2Box").checked;
    clearTimeouts();
    refreshViewTwice();
    updateAi();
}

function refreshViewTwice() {
    refreshView();
    refreshView();
}

function initCheckBoxValues() {
    document.getElementById("comP1Box").checked = isAIByPlayerId[0];
    document.getElementById("comP2Box").checked = isAIByPlayerId[1];
}

async function loadKerasModel() {
    return tf.loadLayersModel("model/model.json");
}

function convertBoardToInput(board) {
    //input shape is [1,8,10,2] -> [batch, height, width, channels (0 = current player, 1 = other player)]
    //value is 1.5 if token is on field, 0 if not
    //an 4dtensor of type float32 with 1*8*10*2 = 160 values is returned
    let inputDepth1 = new Array(numberOfFields * 2);
    let inputDepth2 = new Array(numberOfFields * 2 * boardWide);
    let inputInverse = new Array(numberOfFields * 2);
    let valueCurrentPlayer;
    let valueOtherPlayer;
    let valueCurrentField;
    for (let i = 0; i < numberOfFields; i++) {
        //for each i we have 2 values, one for the current player and one for the other player
        valueCurrentField = board[i];
        if (valueCurrentField === empty) {
            valueCurrentPlayer = 0.0;
            valueOtherPlayer = 0.0;
        } else if (valueCurrentField === currentPlayer) {
            valueCurrentPlayer = 1.5;
            valueOtherPlayer = 0.0;
        } else {
            valueCurrentPlayer = 0.0;
            valueOtherPlayer = 1.5;
        }
        inputDepth1[i * 2] = valueCurrentPlayer;
        inputDepth1[i * 2 + 1] = valueOtherPlayer;
        inputInverse[i * 2] = valueOtherPlayer;
        inputInverse[i * 2 + 1] = valueCurrentPlayer;
    }
    for (let i = 0, indexOffset = 0; i < boardWide; i++) {
        for (let j = 0; j < numberOfFields; j++) {
            inputDepth2[indexOffset + j * 2] = inputInverse[j * 2];
            inputDepth2[indexOffset + j * 2 + 1] = inputInverse[j * 2 + 1];
        }
        if (gapInformation[i] >= 0) {
            inputDepth2[indexOffset + gapInformation[i] * 2 + 1] = 1.5;
        }
        indexOffset += numberOfFields * 2;
    }
    //inputInverse is the same as input, but x is inverted
    /*for (let y = 0; y < boardHeight; ++y) {
        for (let x = 0; x < boardWide; ++x) {
            let index = y * boardWide + x;
            let indexInverse = y * boardWide + (boardWide - 1 - x);
            inputInverse[indexInverse * 2] = input[index * 2];
            inputInverse[indexInverse * 2 + 1] = input[index * 2 + 1];
        }
    }*/
    let inputTensorDepth1 = tf.tensor4d(inputDepth1, [1, boardHeight, boardWide, 2]);
    let inputTensorDepth2 = tf.tensor4d(inputDepth2, [10, boardHeight, boardWide, 2]);
    let inputTensor4d = tf.concat([inputTensorDepth1, inputTensorDepth2], 0);
    return inputTensor4d;
}

function predictMoveEvaluations() {
    let input = convertBoardToInput(board);
    //get prediction as float32 array
    let prediction = kerasAIModel.predict(input).dataSync();
    let predictionAverageMiniMax = new Array(boardWide);

    //calculate average of all predictions for each column
    for (let i = 0; i < boardWide; i++) {
        let maxValueForOpponent = -1;
        let offset = (i + 1) * boardWide;
        for (let j = 0; j < boardWide; j++) {
            if ((gapInformation[j] >= boardWide || (i != j && gapInformation[j] >= 0)) && prediction[offset + j] > maxValueForOpponent) {
                maxValueForOpponent = prediction[offset + j];
            }
        }
        if (maxValueForOpponent > -0.1)
            predictionAverageMiniMax[i] = (prediction[i] + (1.0 - maxValueForOpponent)) / 2; //(best move for current player - (best move for opponent)) / 2 => significantly more stable evaluation
        else
            predictionAverageMiniMax[i] = prediction[i];
    }

    return predictionAverageMiniMax;
}


function updateAi() {
    if (gameIsOver || stopAI)
        return;
    if (isAIByPlayerId[currentPlayer - 1]) {
        let currentTimeMillis = new Date().getTime();
        prediction = predictMoveEvaluations();
        console.log("prediction time: " + (new Date().getTime() - currentTimeMillis));
        //originally i didn't train the model to execute the last winning move, so this is the single thing checked explicity, but it meight also work withouth that check
        let bestMove = -1;
        let bestMoveValue = -1.0;
        let winningMoveFound = false;
        console.log("prediction: " + prediction);
        for (let i = 0; i < boardWide; i++) {
            if (isLegalMove(i)) {
                if (isWinningMove(gapInformation[i])) {
                    bestMove = i;
                    winningMoveFound = true;
                    prediction[i] = 1.0;
                }
                if (winningMoveFound)
                    continue;
                if (prediction[i] > bestMoveValue) {
                    bestMove = i;
                    bestMoveValue = prediction[i];
                }
            }
        }
        if (!winningMoveFound && moveNumber < 8) {
            // at opening a random move with good evaluation is chosen max 0.07 worse than the best move
            let bestAcceptedMoveValue = bestMoveValue - 0.074;
            let acceptedMoves = [];
            for (let i = 0; i < boardWide; i++) {
                if (isLegalMove(i) && prediction[i] >= bestAcceptedMoveValue) {
                    acceptedMoves.push(i);
                }
            }
            savedAIMove = acceptedMoves[Math.floor(Math.random() * acceptedMoves.length)];
        } else {
            savedAIMove = bestMove;
        }
        console.log("move decision: " + savedAIMove);
        let currentDate = new Date();
        let delayTimeNeededForAi = aiDelayTime - (currentDate.getTime() - lastMoveExecutionTime);
        if (delayTimeNeededForAi > refreshBoardFrequence)
            setTimeout(executeLastPredictedMove, delayTimeNeededForAi);
        else
            setTimeout(executeLastPredictedMove, refreshBoardFrequence);
        updateActivationLabelValues(prediction, currentPlayer);
    }
}

function updateActivationLabelValues(values, player) {
    for (let i = 0; i < boardWide; i++) {
        let activationLabel = document.getElementById(i + "_activationLabel");
        if (isLegalMove(i)) {
            activationLabel.innerHTML = Math.round(values[i] * 10) / 10;
        } else {
            activationLabel.innerHTML = "NaN";
        }
        if (player === 1) {
            activationLabel.style.color = p1Color;
        } else {
            activationLabel.style.color = p2Color;
        }
    }
}

function executeLastPredictedMove() {
    if (savedAIMove === -1)
        return;
    executeMove(savedAIMove, true);
}

function refreshBoardParameters() {
    let boardDOM = document.getElementById("board");
    let boardPosition = boardDOM.getBoundingClientRect();
    singleFieldWidth = (boardPosition.width) / boardWide;
    singleFieldHeight = singleFieldWidth;
    boardPositionX = boardPosition.left;
    boardPositionY = boardPosition.top + window.scrollY;
    arrowWide = singleFieldWidth * arrowSizeFactor;
    arrowHeight = singleFieldHeight * arrowSizeFactor;
    console.log(singleFieldWidth + " " + singleFieldHeight);
}

function initBoardView() {
    generateBoardElements();
    refreshView();
}

function generateBoardElements() {
    var boardDOM = document.getElementById("board");
    //get number of children
    var numberOfChildren = boardDOM.childElementCount;
    if (numberOfChildren > 2)
        return;
    //clear all children
    while (boardDOM.firstChild) {
        boardDOM.removeChild(boardDOM.firstChild);
    }
    //add pictures to board
    for (let i = 0; i < numberOfFields; i++) {
        //boardDOM.innerHTML += "<img id=" + i + "_chipYellow src='img/connect_four_chip_cyan.png'></img>";
        //id sheme: v<fieldIndex>_<token>
        boardDOM.innerHTML += "<img id=" + "v" + i + "_" + p1Token + " src='img/connect_four_chip_yellow.png'></img>";
        boardDOM.innerHTML += "<img id=" + "v" + i + "_" + p2Token + " src='img/connect_four_chip_cyan.png'></img>";
        boardDOM.innerHTML += "<img id=" + i + "_background src='img/connect_four_field_background.png'></img>";
        boardDOM.innerHTML += "<img id=" + i + "_foreground src='img/connect_four_field_foreground.png'></img>";
        //boardDOM.innerHTML += "<img id=" + i + "_fieldEmpty src='img/connect_four_field_empty.png'></img>";
    }
    for (let i = 0; i < boardWide; i++) {
        //id sheme: a<fieldIndex>_<token>
        boardDOM.innerHTML += "<img id=" + "a" + i + "_" + p1Token + " src='img/arrowYellow.png'></img>";
        boardDOM.innerHTML += "<img id=" + "a" + i + "_" + p2Token + " src='img/arrowCyan.png'></img>";
        boardDOM.innerHTML += "<img id=" + "a" + i + "_" + highlightArrow + " src='img/arrowHighlight.png'></img>";
        boardDOM.innerHTML += "<img id=" + "a" + i + "_" + empty + " src='img/arrowGrey.png'></img>";
    }
    //place gap markers
    for (let i = 0; i < boardWide; i++) {
        boardDOM.innerHTML += "<div id=" + i + "_gapMarker class='gapmarker'></div>";
    }
    //place activation labels
    for (let i = 0; i < boardWide; i++) {
        boardDOM.innerHTML += "<div id=" + i + "_activationLabel class='activationlabel'></div>";
    }
}

function getLeftDistanceByFieldNumber(fieldNumber) {
    return boardPositionX + (fieldNumber % boardWide) * (singleFieldWidth);
}

function getTopDistanceByFieldNumber(fieldNumber) {
    return boardPositionY + (Math.floor(fieldNumber / boardWide) + 1) * (singleFieldHeight);
}

function getLeftDistanceByFieldNumberForArrow(fieldNumber) {
    return boardPositionX + (fieldNumber % boardWide) * (singleFieldWidth) + (singleFieldWidth - (singleFieldWidth * arrowSizeFactor)) / 2;
}

function getTopDistanceByFieldNumberForArrow() {
    return boardPositionY + singleFieldHeight * arrowRowOffset;
}

function placeElement(element, x, y, width, height, zIndex, visibility) {
    element.style.position = "absolute";
    element.style.top = y + "px";
    element.style.left = x + "px";
    element.style.width = width + "px";
    element.style.height = height + "px";
    element.style.zIndex = zIndex;
    element.style.visibility = visibility ? "visible" : "hidden";
}

function initMoveHistory() {
    for (let i = 0; i < moveHistory.length; i++) {
        moveHistory[i] = -1;
    }
}

function initBoard() {
    for (let i = 0; i < numberOfFields; i++) {
        board[i] = empty;
    }
    for (let i = 0; i < boardWide; i++) {
        gapInformation[i] = numberOfFields - (boardWide - i);
    }
    for (let i = 0; i < animationBoard.length; i++) {
        animationBoard[i] = empty;
    }
}

//function not used in this version
function onKeyPress(event) {
    if (event.ctrlKey) {
        if (event.key === 'z' || event.key === 'Z') {
            onUndoMoveRequest();
        } else if (event.key === 'y' || event.key === 'Y') {
            onReExecuteMove();
        }
    } else if (event.key === 'n' || event.key === 'N') {
        if (moveNumber === 0)
            return;
        nKeyCount++;
        if (nKeyCount === 2)
            initGame();
        else
            setTimeout(resetNKeyCount, nKeyCountResetDelayTime);
    } else if (event.key >= '1' && event.key <= String.fromCharCode('0'.charCodeAt() + boardWide)) {
        onMoveExecutionRequest(event.key.charCodeAt() - '1'.charCodeAt(), true);
    }
}

function resetNKeyCount() {
    nKeyCount = 0;
}

function onUndoMoveRequest() {
    if (moveNumber > 0)
        undoMove();
}

function onMouseClick(event) {
    if (calculateRowOfPosition(event) > boardHeight)
        return;
    onMoveExecutionRequest(calculateGapOfPosition(event), true);
}

function disableAnimationMode() {
    inAninmationMode = false;
    p1InAnimationMode = false;
    p2InAnimationMode = false;
}

function onReExecuteMove() {
    if (inAninmationMode) {
        disableAnimationMode();
        refreshView(); //vorspulen der aktuell laufenden animation
    } else if (moveHistory[moveNumber] !== -1) {
        onMoveExecutionRequest(moveHistory[moveNumber], false); //ehemaligen Zug ausführen
    }
}

function onMoveExecutionRequest(move, activateAnimationMode) {
    date = new Date();
    let time = date.getTime();
    if (lastMoveExecutionTime + delayToNextMove >= time || p1InAnimationMode && p2InAnimationMode || time - initFinishTime < initFinishTimeMoveAcceptanceDelay) {
        return; //Doppelklick auf die selbe Spalte wird ignoriert barrierefreiheit, zitternde hände
    }
    if (gameIsOver) {
        alert("The game is over! Start an new game to continue.");
        return;
    }
    if (isAIByPlayerId[currentPlayer - 1])
        return;
    if (!isLegalMove(move)) {
        alert("This gap is full!");
        return;
    }
    executeMove(move, activateAnimationMode); //Führt den Zug aus, startet die Animation und stellt einen Sieger fest
}

function isLegalMove(move) {
    return board[move] === empty;
}

function executeMove(move, activateAnimationMode) {
    let moveFieldIndex = gapInformation[move];
    board[moveFieldIndex] = currentPlayer;
    gameIsOver = isWinningMove(moveFieldIndex);
    gapInformation[move] -= boardWide;
    moveHistory[moveNumber++] = move;
    if (!gameIsOver) {
        if (currentPlayer === 1) {
            if (activateAnimationMode)
                activateP1AnimationMode(move);
            currentPlayer = 2;
            currentPlayerText = p2MoveText;
        } else {
            if (activateAnimationMode)
                activateP2AnimationMode(move);
            currentPlayer = 1;
            currentPlayerText = p1MoveText;
        }
        if (moveNumber >= numberOfFields) {
            gameIsOver = true;
            currentPlayerText = "The game ends draw!"
        }
    } else {
        if (currentPlayer === 1) {
            activateP1AnimationMode(move);
            currentPlayerText = "Yellow wins!";
        } else {
            activateP2AnimationMode(move);
            currentPlayerText = "Cyan wins!";
        }
    }
    lastMoveGap = move;
    if (activateAnimationMode || gameIsOver) {
        startAnimation();
    } else {
        inAninmationMode = false;
        refreshView();
    }
    if (activateAnimationMode)
        lastMoveExecutionTime = date.getTime();
    setTimeout(updateAi, 5);
}

function undoMove() {
    let lastMove = moveHistory[--moveNumber];
    let moveFieldIndex = gapInformation[lastMove] + bottom;
    board[moveFieldIndex] = empty;
    gapInformation[lastMove] = moveFieldIndex;
    lastMoveGap = moveNumber > 0 ? moveHistory[moveNumber - 1] : -1;
    if ((currentPlayer === 1) !== gameIsOver) {
        currentPlayer = 2;
        currentPlayerText = p2MoveText;
    } else {
        currentPlayer = 1;
        currentPlayerText = p1MoveText;
    }
    gameIsOver = false;
    inAninmationMode = false;
    p1InAnimationMode = false;
    p2InAnimationMode = false;
    refreshView();
}

function activateP1AnimationMode(move) {
    p1NextAnimationPosition = move;
    p1LastAnimationTime = 0; //January 1, 1970 führt zu sofortiger aktuallisierung bei nächster animationsfortschrittsüberprüfung, welche durch executemove ebenfalls eingeleitet wird
    p1InAnimationMode = true;
}

function activateP2AnimationMode(move) {
    p2NextAnimationPosition = move;
    p2LastAnimationTime = 0; //January 1, 1970 führt zu sofortiger aktuallisierung bei nächster animationsfortschrittsüberprüfung, welche durch executemove ebenfalls eingeleitet wird
    p2InAnimationMode = true;
}

function startAnimation() {
    inAninmationMode = true;
    onTimerRefresh();
    refreshTextView();
}

async function onTimerRefresh() //leitet die aktuallisierung der oberfläche ein und entscheidet, ob onTimerRefresh nach einer bestimmten zeit erneut aufgerufen wird
{
    if (!inAninmationMode) {
        refreshTextView();
        return; //kann beispielsweise eintreten, Falls die animation läuft und der spieler währenddessen ein neues spiel startet
    }
    date = new Date();
    if (p1InAnimationMode && checkRefreshAnimationPlayer(p1LastAnimationTime, p1NextAnimationPosition)) {
        p1InAnimationMode = refreshAnimationPlayer(p1NextAnimationPosition, p1Token);
        p1NextAnimationPosition += bottom;
        p1LastAnimationTime = date.getTime();

    }
    if (p2InAnimationMode && checkRefreshAnimationPlayer(p2LastAnimationTime, p2NextAnimationPosition)) {
        p2InAnimationMode = refreshAnimationPlayer(p2NextAnimationPosition, p2Token);
        p2NextAnimationPosition += bottom;
        p2LastAnimationTime = date.getTime();
    }
    inAninmationMode = p1InAnimationMode || p2InAnimationMode;
    if (gameIsOver && !inAninmationMode) {
        setTimeout(onTimerRefresh, gameEndFrequence);
    } else {
        setTimeout(onTimerRefresh, refreshBoardFrequence); //Anforderung der Wiederholung dieser Methode in refreshBoardFrequence Millisekunden
    }
}


function checkRefreshAnimationPlayer(lastAnimationTime, nextAnimationPosition) {
    if (nextAnimationPosition - boardWide < boardWide) { //arrow animation?
        return date.getTime() - lastAnimationTime >= animationTimeFrequence + animationTimeArrowOffset;
    } else {
        return date.getTime() - lastAnimationTime >= animationTimeFrequence;
    }

}

function refreshAnimationPlayer(nextAnimationPosition, token) {
    let currentAnimationPosition = nextAnimationPosition - bottom;
    animationBoard[nextAnimationPosition] = token;
    refreshSingleTokenView(nextAnimationPosition);
    if (currentAnimationPosition >= 0) {
        if (currentAnimationPosition < boardWide && currentAnimationPosition === lastMoveGap) {
            animationBoard[currentAnimationPosition] = highlightArrow;
        } else {
            animationBoard[currentAnimationPosition] = empty;
        }
        refreshSingleTokenView(currentAnimationPosition);
    } else {
        for (let arrowIndex = 0; arrowIndex < boardWide; arrowIndex++) {
            if (animationBoard[arrowIndex] === highlightArrow) {
                animationBoard[arrowIndex] = empty;
                refreshSingleTokenView(arrowIndex);
            }
        }
        return true; //die animation eines gerade eingeworfenen chips wird immer fortgesetzt, da dieser erst im zweiten Animationsschritt auf dem Spielfeld erscheint
    }
    return board[currentAnimationPosition] !== token;
}

function calculateGapOfPosition(event) { //rechnet die Mausposition in eine Spalte um
    //refreshBoardParameters();
    let relativeX = event.clientX - boardPositionX;
    if (relativeX < 0)
        relativeX = 0;
    let clickedGap = Math.floor(relativeX / singleFieldWidth);
    if (clickedGap >= boardWide)
        clickedGap = boardWide - 1;
    return clickedGap;
}

function calculateRowOfPosition(event) {
    let relativeY = event.clientY - boardPositionY;
    if (relativeY < 0)
        relativeY = 0;
    let clickedRow = Math.floor(relativeY / singleFieldHeight);
    return clickedRow;
}

function refreshView() {
    refreshBoardParameters();
    for (let currentFieldIndex = 0; currentFieldIndex < animationBoard.length; currentFieldIndex++) {
        if (currentFieldIndex < boardWide) {
            if (currentFieldIndex === lastMoveGap) {
                animationBoard[currentFieldIndex] = highlightArrow;
            } else {
                animationBoard[currentFieldIndex] = empty;
            }
        } else {
            let i = currentFieldIndex - bottom;
            animationBoard[currentFieldIndex] = board[i];
            let leftDistance = getLeftDistanceByFieldNumber(i);
            let topDistance = getTopDistanceByFieldNumber(i);
            placeElement(document.getElementById(i + "_background"), leftDistance, topDistance, singleFieldWidth, singleFieldHeight, 1, true);
            placeElement(document.getElementById(i + "_foreground"), leftDistance, topDistance, singleFieldWidth, singleFieldHeight, 3, true);
        }
        refreshSingleTokenView(currentFieldIndex);
    }
    let pixelHeight = singleFieldHeight * (boardHeight + 1.5)
    let gapMarkerHeight = singleFieldHeight * (boardHeight + 1)
    document.getElementById("board").style.height = pixelHeight + "px";
    refreshGapMarkers(gapMarkerHeight);
    refreshActivationLabels(gapMarkerHeight);
    refreshTextView();
}

function refreshGapMarkers(pixelHeight) {
    for (let markerId = 0; markerId < boardWide; markerId++) {
        let guiObject = document.getElementById(markerId + "_gapMarker");
        placeElement(guiObject, getLeftDistanceByFieldNumber(markerId), getTopDistanceByFieldNumber(0) - singleFieldHeight, singleFieldWidth, pixelHeight, 4, true);
    }
}

function refreshActivationLabels(pixelHeight) {
    for (let i = 0; i < boardWide; i++) {
        let activationLabel = document.getElementById(i + "_activationLabel");
        placeElement(activationLabel, getLeftDistanceByFieldNumber(i), boardPositionY + pixelHeight + singleFieldHeight * activationLabelRowOffset, singleFieldWidth, singleFieldHeight / 2, 4, true);
        if (prediction === null) {
            activationLabel.innerHTML = "???";
            activationLabel.style.color = p2Color;
        }
    }
}

function synchronizeMarkerSize(marker) {
    marker.style.height = (singleFieldHeight * (boardHeight + (1 - arrowRowOffset))) + "px";
}

function refreshTextView() {
    document.getElementById("currentPlayerText").innerHTML = currentPlayerText;
    if (gameIsOver) {
        if (!inAninmationMode && !winMassageShown) {
            alert(currentPlayerText);
            winMassageShown = true;
        }
    } else {
        winMassageShown = false;
    }

}

function refreshSingleTokenView(fieldIndex) {
    let guiObject;
    if (fieldIndex < boardWide) //arrows!
    {
        let arrow = animationBoard[fieldIndex];
        for (let currentArrow = 0; currentArrow <= highlightArrow; currentArrow++) {
            guiObject = document.getElementById("a" + fieldIndex + "_" + currentArrow);
            //synchronizePosition(guiObject, fieldIndex);
            placeElement(guiObject, getLeftDistanceByFieldNumberForArrow(fieldIndex), getTopDistanceByFieldNumberForArrow(fieldIndex), arrowWide, arrowHeight, 1, currentArrow === arrow);
            //console.log("Arrow | " + "width: " + singleFieldWidth + " height: " + singleFieldHeight + " fieldIndex: " + fieldIndex);
        }
    } else { //chips!
        let token = animationBoard[fieldIndex];
        for (let currentToken = p1Token; currentToken <= p2Token; currentToken++) {
            guiObject = document.getElementById("v" + (fieldIndex - boardWide) + "_" + currentToken);
            placeElement(guiObject, getLeftDistanceByFieldNumber(fieldIndex - boardWide), getTopDistanceByFieldNumber(fieldIndex - boardWide), singleFieldWidth, singleFieldHeight, 2, currentToken === token);
            //log width, height, and fieldIndex
            //console.log("Chip | " + "width: " + singleFieldWidth + " height: " + singleFieldHeight + " fieldIndex: " + fieldIndex);
        }
    }
}

function isWinningMove(move) {
    return checkRow(winDistance, right, move, false, true, false) ||
        checkRow(winDistance, bottomRight, move, true, true, false) ||
        checkRow(winDistance, bottom, move, true, false, false) ||
        checkRow(winDistance, bottomLeft, move, true, false, true);
}

function checkRow(distance, direction, position, limitBottom, limitRight, limitLeft) { //limit gibt an, ob in die gegebene richtung nach dem Spielfeldende zu prüfen ist
    //distance gibt die größe der reihe an, welche gleiche tokens haben muss (bei 4 gewinnt gilt: distance==4)
    let maxDistance = distance - 1; //maxDistance beschreibt die größt mögliche bewegungsweite, welche in richtung Direction zulässig ist, ohne dass der spielfeldrand überschritten wird
    let maxNegativeDistance = distance - 1; //das gleiche wie maxDistanz, nur in die entgegengesetzte Richtung
    let currentDistance;
    let currentGapRow;
    if (limitBottom) {
        currentGapRow = Math.floor(position / boardWide); //aktuelle zeile (javascript beherscht scheinbar keine integer division, daher ist eine nachträglche abrundung erforderlich)
        currentDistance = boardHeight - (currentGapRow + 1); //anzahl felder bis zum unteren Spielfeldrand
        if (currentDistance < maxDistance)
            maxDistance = currentDistance;
        if (currentGapRow < maxNegativeDistance)
            maxNegativeDistance = currentGapRow;
    }
    if (limitRight || limitLeft) {
        currentGapRow = position % boardWide; //aktuelle Spalte
        currentDistance = boardWide - (currentGapRow + 1);
        if (limitRight) {
            if (currentDistance < maxDistance)
                maxDistance = currentDistance;
            if (currentGapRow < maxNegativeDistance)
                maxNegativeDistance = currentGapRow;
        } else {
            if (currentDistance < maxNegativeDistance)
                maxNegativeDistance = currentDistance;
            if (currentGapRow < maxDistance)
                maxDistance = currentGapRow;
        }
    }
    let count = 1;
    let token = board[position];
    if (token == empty)
        token = currentPlayer;
    let currentPosition = position + direction;
    for (let a = 1; a <= maxDistance; a++, currentPosition += direction) { //durchläuft das Spielfeld in direction bis ein token != playerId und zählt anzahl treffer==count
        if (board[currentPosition] !== token)
            break;
        count++;
    }
    currentPosition = position - direction;
    for (let a = 1; a <= maxNegativeDistance; a++, currentPosition -= direction) { //durchläuft das Spielfeld in -direction bis ein token != playerId und zählt anzahl treffer==count
        if (board[currentPosition] !== token)
            break;
        count++;
    }
    return count >= distance; //gibt true zurück, Falls distance viele gleiche in einer Reihe sind
}