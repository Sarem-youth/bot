#property copyright "Your Company"
#property link      "https://yourcompany.com"
#property version   "1.00"
#property strict

// Include required libraries
#include <Trade\Trade.mqh>
#include <Files\FilePipe.mqh>

// Initialize global variables
CTrade trade;
CFilePipe pipe;

// Configuration
input string PIPE_NAME = "mt5_bridge";  // Named pipe for communication
input int MAGIC_NUMBER = 234000;        // Magic number for trade identification

// Global variables for communication
string g_last_error = "";
bool g_is_connected = false;

// Initialize Expert Advisor
int OnInit() {
    // Configure trade settings
    trade.SetExpertMagicNumber(MAGIC_NUMBER);
    trade.SetMarginMode();
    trade.SetTypeFillingBySymbol(_Symbol);
    
    // Initialize communication pipe
    if(!InitializeCommunication()) {
        Print("Failed to initialize communication: ", g_last_error);
        return INIT_FAILED;
    }
    
    Print("MT5 Bridge initialized successfully");
    return(INIT_SUCCEEDED);
}

// Initialize communication channel
bool InitializeCommunication() {
    if(!pipe.Open(PIPE_NAME, FILE_READ|FILE_WRITE)) {
        g_last_error = "Failed to open communication pipe";
        return false;
    }
    
    g_is_connected = true;
    return true;
}

// Main trading logic
void OnTick() {
    if(!g_is_connected) return;
    
    // Check for new commands
    CheckForCommands();
    
    // Send tick data
    SendTickData();
}

// Check for incoming commands
void CheckForCommands() {
    string command = "";
    
    if(pipe.ReadString(command)) {
        if(StringLen(command) > 0) {
            HandleCommand(command);
        }
    }
}

// Handle incoming commands
void HandleCommand(string command) {
    string parts[];
    int count = StringSplit(command, ',', parts);
    
    if(count < 1) return;
    
    string cmd = parts[0];
    
    if(cmd == "OPEN") {
        if(count < 7) return;
        
        string symbol = parts[1];
        ENUM_ORDER_TYPE type = (parts[2] == "BUY") ? ORDER_TYPE_BUY : ORDER_TYPE_SELL;
        double volume = StringToDouble(parts[3]);
        double price = StringToDouble(parts[4]);
        double sl = StringToDouble(parts[5]);
        double tp = StringToDouble(parts[6]);
        
        ExecuteOrder(symbol, type, volume, price, sl, tp);
    }
    // ... rest of command handling ...
}

// Send current tick data
void SendTickData() {
    MqlTick tick;
    if(SymbolInfoTick(_Symbol, tick)) {
        string message = StringFormat(
            "TICK,%s,%.5f,%.5f,%d,%d",
            _Symbol,
            tick.bid,
            tick.ask,
            tick.volume,
            tick.time
        );
        pipe.WriteString(message);
    }
}

// Execute trading order
bool ExecuteOrder(string symbol, ENUM_ORDER_TYPE type, double volume,
                 double price, double sl, double tp) {
    return trade.PositionOpen(
        symbol,
        type,
        volume,
        price,
        sl,
        tp,
        "Python Bot"
    );
}

// Clean up on Expert Advisor removal
void OnDeinit(const int reason) {
    if(g_is_connected) {
        pipe.Close();
    }
    g_is_connected = false;
    Print("MT5 Bridge shutdown completed");
}
