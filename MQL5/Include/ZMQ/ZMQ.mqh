#property copyright "MetaTrader 5"
#property link      "https://www.mql5.com"
#property strict

// ZMQ Version
#define ZMQ_VERSION_MAJOR 4
#define ZMQ_VERSION_MINOR 3
#define ZMQ_VERSION_PATCH 2

// Socket types
#define ZMQ_PAIR 0
#define ZMQ_PUB 1
#define ZMQ_SUB 2
#define ZMQ_REQ 3
#define ZMQ_REP 4
#define ZMQ_DEALER 5
#define ZMQ_ROUTER 6
#define ZMQ_PULL 7
#define ZMQ_PUSH 8

// Import ZMQ DLL functions
#import "libzmq.dll"
   void* zmq_ctx_new();
   int zmq_ctx_destroy(void* context);
   void* zmq_socket(void* context, int type);
   int zmq_close(void* socket);
   int zmq_bind(void* socket, const string endpoint);
   int zmq_connect(void* socket, const string endpoint);
   int zmq_send(void* socket, const uchar& data[], int length, int flags);
   int zmq_recv(void* socket, uchar& data[], int length, int flags);
   int zmq_setsockopt(void* socket, int option_name, const void* option_value, size_t option_len);
#import

// ZMQ Context class
class ZmqContext {
private:
    void* m_context;

public:
    ZmqContext() : m_context(NULL) {}
    ~ZmqContext() { Destroy(); }
    
    bool Create() {
        m_context = zmq_ctx_new();
        return m_context != NULL;
    }
    
    void Destroy() {
        if(m_context != NULL) {
            zmq_ctx_destroy(m_context);
            m_context = NULL;
        }
    }
    
    void* Handle() { return m_context; }
};

// ZMQ Socket class
class ZmqSocket {
private:
    void* m_socket;
    
public:
    ZmqSocket() : m_socket(NULL) {}
    ~ZmqSocket() { Destroy(); }
    
    bool Create(ZmqContext& context, int type) {
        m_socket = zmq_socket(context.Handle(), type);
        return m_socket != NULL;
    }
    
    void Destroy() {
        if(m_socket != NULL) {
            zmq_close(m_socket);
            m_socket = NULL;
        }
    }
    
    bool Bind(string endpoint) {
        return zmq_bind(m_socket, endpoint) == 0;
    }
    
    bool Connect(string endpoint) {
        return zmq_connect(m_socket, endpoint) == 0;
    }
    
    bool Send(string data, int flags = 0) {
        uchar data_array[];
        StringToCharArray(data, data_array);
        return zmq_send(m_socket, data_array, ArraySize(data_array)-1, flags) >= 0;
    }
    
    bool Receive(string& data, int flags = 0) {
        uchar data_array[];
        ArrayResize(data_array, 1024);  // Adjust buffer size as needed
        int length = zmq_recv(m_socket, data_array, ArraySize(data_array)-1, flags);
        if(length >= 0) {
            data_array[length] = 0;  // Null terminate
            data = CharArrayToString(data_array);
            return true;
        }
        return false;
    }
    
    bool SetReceiveTimeout(int timeout) {
        return zmq_setsockopt(m_socket, 3, &timeout, 4) == 0;  // ZMQ_RCVTIMEO = 3
    }
    
    bool SetSendTimeout(int timeout) {
        return zmq_setsockopt(m_socket, 4, &timeout, 4) == 0;  // ZMQ_SNDTIMEO = 4
    }
};
