#ifndef __CDMA_REG_DEF_H__
#define __CDMA_REG_DEF_H__

// cdma reg id definition
#define CDMA_ID_INTR_EN                         {0, 1}
#define CDMA_ID_STRIDE_ENABLE                   {1, 1}
#define CDMA_ID_NCHW_COPY                       {2, 1}
#define CDMA_ID_CMD_TYPE                        {4, 4}
#define CDMA_ID_CMD_SPECIAL_FUNCTION            {8, 3}
#define CDMA_ID_FILL_CONSTANT_EN                {11, 1}
#define CDMA_ID_SRC_DATA_FORMAT                 {12, 4} // NOT USED IN DMA_SEND

// cdma_send, cdma_lossy_compress, cdma_loss_decompress
#define CDMA_ID_SEND_SRC_ADDR_H13               {16, 13}
#define CDMA_ID_SEND_PSUM_OP                    {29, 3}
#define CDMA_ID_MSG_ID                          {32, 9}
#define CDMA_ID_MSG_CNT                         {48, 7}
#define CMDA_ID_SEND_NSTRIDE                    {32, 32}
#define CMDA_ID_SEND_CONSTANT_VALUE             {32, 32} // when CMD_SPECIAL_FUNCTION==FILL CONSTANT
#define CMDA_ID_SEND_CSTRIDE                    {64, 32}
#define CMDA_ID_SEND_HSTRIDE                    {96, 32}
#define CMDA_ID_SEND_NSIZE                      {128, 16}
#define CMDA_ID_SEND_CSIZE                      {144, 16}
#define CMDA_ID_SEND_HSIZE                      {160, 32}
#define CMDA_ID_SEND_WSIZE                      {192, 32}
#define CDMA_ID_SEND_SRC_ADDR_L32               {224, 32}

// cmda_recv
#define CDMA_ID_RECV_DST_ADDR_H13               {16, 13}
#define CDMA_ID_RECV_REDUCE_OP                  {29, 3}
#define CMDA_ID_RECV_NSTRIDE                    {32, 32}
#define CMDA_ID_RECV_CSTRIDE                    {64, 32}
#define CMDA_ID_RECV_HSTRIDE                    {96, 32}
#define CMDA_ID_RECV_NSIZE                      {128, 16}
#define CMDA_ID_RECV_CSIZE                      {144, 16}
#define CMDA_ID_RECV_HSIZE                      {160, 32}
#define CMDA_ID_RECV_WSIZE                      {192, 32}
#define CDMA_ID_RECV_DST_ADDR_L32               {224, 32}

#define CDMA_ID_RECV_DATA_FORMAT                {11, 3} //RESERVED, only for cmodel
#define CDMA_ID_RECV_SEND_TSK_INDEX             {14, 2} //RESERVED, for model tool

//cdma_p2p_send(cdma_general)
#define CDMA_ID_GENERAL_SRC_ADDR_H13            {16, 13}
#define CDMA_ID_GENERAL_DST_ADDR_H13            {32, 13}
#define CDMA_ID_GENERAL_CMD_LENGTH              {64, 32}
#define CDMA_ID_GENERAL_SRC_ADDR_L32            {96, 32}
#define CDMA_ID_GENERAL_CONST_VAL               {96, 32}
#define CDMA_ID_GENERAL_DST_ADDR_L32            {128, 32}

#define CDMA_ID_GENERAL_SRC_CHIPID              {45, 4} //RESERVED, only for cmodel
#define CDMA_ID_GENERAL_DST_CHIPID              {49, 4} //RESERVED, only for cmodel
#define CDMA_ID_BATCH_SEND_RECV_GENERAL_FIRST_LOOP      {53, 1} //RESERVED, only for cmodel
#define CDMA_ID_BATCH_SEND_RECV_GENERAL_LAST_LOOP       {54, 1} //RESERVED, only for cmodel

//cdma_read/cdma_write
#define CDMA_ID_READ_SRC_ADDR_H13              {16, 13}
#define CDMA_ID_READ_DST_ADDR_H13              {32, 13}
#define CMDA_ID_READ_SRC_NSTRIDE               {64, 32}
#define CMDA_ID_READ_CONST_VALUE               {64, 32}
#define CMDA_ID_READ_SRC_CSTRIDE               {96, 32}
#define CMDA_ID_READ_SRC_HSTRIDE               {128, 32}
#define CMDA_ID_READ_DST_NSTRIDE               {160, 32}
#define CMDA_ID_READ_DST_CSTRIDE               {192, 32}
#define CMDA_ID_READ_DST_HSTRIDE               {224, 32}
#define CMDA_ID_READ_SRC_NSIZE                 {256, 16}
#define CMDA_ID_READ_SRC_CSIZE                 {272, 16}
#define CMDA_ID_READ_SRC_HSIZE                 {288, 32}
#define CMDA_ID_READ_SRC_WSIZE                 {320, 32}
#define CMDA_ID_READ_DST_NSIZE                 {352, 16}
#define CMDA_ID_READ_DST_CSIZE                 {368, 16}
#define CMDA_ID_READ_DST_HSIZE                 {384, 32}
#define CMDA_ID_READ_DST_WSIZE                 {416, 32}
#define CDMA_ID_READ_SRC_ADDR_L32              {448, 32}
#define CDMA_ID_READ_DST_ADDR_L32              {480, 32}

#define CDMA_ID_READ_SRC_CHIPID                {45, 4} //RESERVED, only for cmodel
#define CDMA_ID_READ_DST_CHIPID                {49, 4} //RESERVED, only for cmodel
#define CDMA_ID_BATCH_SEND_RECV_FIRST_LOOP     {8, 1} //RESERVED, only for cmodel
#define CDMA_ID_BATCH_SEND_RECV_LAST_LOOP      {9, 1} //RESERVED, only for cmodel

// cdma tcp send/recv
#define CDMA_ID_TCP_OWN                        {1, 1}
#define CDMA_ID_TCP_FD                         {2, 1}
#define CDMA_ID_TCP_LD                         {3, 1}
#define CDMA_ID_TCP_BUFFER_LENGTH              {8, 16}
#define CDMA_ID_TCP_FRAME_LENGTH               {32, 16}
#define CDMA_ID_TCP_BUFFER_ADDR_L32            {64, 32}
#define CDMA_ID_TCP_BUFFER_ADDR_H13            {96, 13}

#define CDMA_ID_TCP_SEND_CHIPID                {48, 4} //RESERVED, only for cmodel
#define CDMA_ID_TCP_RECV_CHIPID                {52, 4} //RESERVED, only for cmodel


#define CDMA_PACK_ID_INTR_EN(val)                         {CDMA_ID_INTR_EN             ,(val)}
#define CDMA_PACK_ID_STRIDE_ENABLE(val)                   {CDMA_ID_STRIDE_ENABLE       ,(val)}
#define CDMA_PACK_ID_NCHW_COPY(val)                       {CDMA_ID_NCHW_COPY           ,(val)}
#define CDMA_PACK_ID_CMD_TYPE(val)                        {CDMA_ID_CMD_TYPE            ,(val)}
#define CDMA_PACK_ID_CMD_SPECIAL_FUNCTION(val)            {CDMA_ID_CMD_SPECIAL_FUNCTION,(val)}
#define CDMA_PACK_ID_FILL_CONSTANT_EN(val)                {CDMA_ID_FILL_CONSTANT_EN    ,(val)}
#define CDMA_PACK_ID_SRC_DATA_FORMAT(val)                 {CDMA_ID_SRC_DATA_FORMAT     ,(val)}
#define CDMA_PACK_ID_SEND_SRC_ADDR_H13(val)               {CDMA_ID_SEND_SRC_ADDR_H13   ,(val)}
#define CDMA_PACK_ID_SEND_PSUM_OP(val)                    {CDMA_ID_SEND_PSUM_OP        ,(val)}
#define CMDA_PACK_ID_SEND_NSTRIDE(val)                    {CMDA_ID_SEND_NSTRIDE        ,(val)}
#define CMDA_PACK_ID_SEND_CONSTANT_VALUE(val)             {CMDA_ID_SEND_CONSTANT_VALUE ,(val)}
#define CMDA_PACK_ID_SEND_CSTRIDE(val)                    {CMDA_ID_SEND_CSTRIDE        ,(val)}
#define CMDA_PACK_ID_SEND_HSTRIDE(val)                    {CMDA_ID_SEND_HSTRIDE        ,(val)}
#define CMDA_PACK_ID_SEND_NSIZE(val)                      {CMDA_ID_SEND_NSIZE          ,(val)}
#define CMDA_PACK_ID_SEND_CSIZE(val)                      {CMDA_ID_SEND_CSIZE          ,(val)}
#define CMDA_PACK_ID_SEND_HSIZE(val)                      {CMDA_ID_SEND_HSIZE          ,(val)}
#define CMDA_PACK_ID_SEND_WSIZE(val)                      {CMDA_ID_SEND_WSIZE          ,(val)}
#define CDMA_PACK_ID_SEND_SRC_ADDR_L32(val)               {CDMA_ID_SEND_SRC_ADDR_L32   ,(val)}
#define CDMA_PACK_ID_RECV_DST_ADDR_H13(val)               {CDMA_ID_RECV_DST_ADDR_H13   ,(val)}
#define CDMA_PACK_ID_RECV_REDUCE_OP(val)                  {CDMA_ID_RECV_REDUCE_OP      ,(val)}
#define CMDA_PACK_ID_RECV_NSTRIDE(val)                    {CMDA_ID_RECV_NSTRIDE        ,(val)}
#define CMDA_PACK_ID_RECV_CSTRIDE(val)                    {CMDA_ID_RECV_CSTRIDE        ,(val)}
#define CMDA_PACK_ID_RECV_HSTRIDE(val)                    {CMDA_ID_RECV_HSTRIDE        ,(val)}
#define CMDA_PACK_ID_RECV_NSIZE(val)                      {CMDA_ID_RECV_NSIZE          ,(val)}
#define CMDA_PACK_ID_RECV_CSIZE(val)                      {CMDA_ID_RECV_CSIZE          ,(val)}
#define CMDA_PACK_ID_RECV_HSIZE(val)                      {CMDA_ID_RECV_HSIZE          ,(val)}
#define CMDA_PACK_ID_RECV_WSIZE(val)                      {CMDA_ID_RECV_WSIZE          ,(val)}
#define CDMA_PACK_ID_RECV_DST_ADDR_L32(val)               {CDMA_ID_RECV_DST_ADDR_L32   ,(val)}


// cdma control reg definition
#define CDMA_ID_CFG_DES_ENABLE                 {0, 1} // CDMA_CSR_0
#define CDMA_ID_CFG_CMD_ID_RESET               {1, 1}
#define CDMA_ID_CFG_DES_CLR                    {2, 1}

#define CDMA_ID_CFG_BASE_MSGID                 {128, 9} // CDMA_CSR_4
#define CDMA_ID_CFG_IST_FIFO_DEPTH             {320, 7} // fifo depth
#define CDMA_ID_CFG_IST_SEND_FIFO_DEPTH        {328, 7} // send fifo depth nouse
#define CDMA_ID_CFG_IST_RECV_FIFO_DEPTH        {336, 7} // recv fifo depth nouse
#define CDMA_ID_CFG_CURRENT_SEND_CMD_ID        {544, 24} // CDMA_CSR_15, h44
#define CDMA_ID_CFG_CURRENT_RCV_CMD_ID         {576, 24} // CDMA_CSR_16, h48
#define CDMA_ID_BASE_ADDR_ID0                  {736, 32} // CDMA_CSR_21, h5c


#endif // __CDMA_REG_DEF_H__