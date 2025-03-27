#ifndef __GDMA_REG_DEF_H__
#define __GDMA_REG_DEF_H__


// gdma reg id defines
#define GDMA_ID_INTR_EN                           {0, 1}
#define GDMA_ID_STRIDE_ENABLE                     {1, 1}
#define GDMA_ID_NCHW_COPY                         {2, 1}
#define GDMA_ID_CMD_SHORT                         {3, 1}
#define GDMA_ID_CACHE_EN                          {4, 1}
#define GDMA_ID_CACHE_FLUSH                       {5, 1}
#define GDMA_ID_CMD_TYPE                          {32, 5}
#define GDMA_ID_CMD_SPECIAL_FUNCTION              {37, 3}
#define GDMA_ID_FILL_CONSTANT_EN                  {40, 1}
#define GDMA_ID_SRC_DATA_FORMAT                   {41, 4}
#define GDMA_ID_MASK_DATA_FORMAT                  {45, 4}  // Also for index
#define GDMA_ID_RAND_MASK_INTER_STATE             {45, 1}
#define GDMA_ID_RAND_MASK_INIT_SEED               {46, 1}
#define GDMA_ID_CMD_ID_DEP                        {64, 20}
#define GDMA_ID_DEP_ID_EN                         {84, 1}
#define GDMA_ID_MSG_ID                            {96, 9}
#define GDMA_ID_MSG_CNT                           {105, 6}
#define GDMA_ID_CONSTANT_VALUE                    {96, 32}
#define GDMA_ID_SRC_NSTRIDE                       {128, 32}
#define GDMA_ID_SRC_CSTRIDE                       {160, 32}
#define GDMA_ID_SRC_HSTRIDE                       {192, 32}
#define GDMA_ID_SRC_WSTRIDE                       {224, 32}
#define GDMA_ID_DST_NSTRIDE                       {256, 32}
#define GDMA_ID_DST_CSTRIDE                       {288, 32}
#define GDMA_ID_DST_HSTRIDE                       {320, 32}
#define GDMA_ID_DST_WSTRIDE                       {352, 32}
#define GDMA_ID_SRC_NSIZE                         {384, 16}
#define GDMA_ID_SRC_CSIZE                         {400, 16}
#define GDMA_ID_SRC_HSIZE                         {416, 16}
#define GDMA_ID_SRC_WSIZE                         {432, 16}
#define GDMA_ID_DST_NSIZE                         {448, 16}
#define GDMA_ID_DST_CSIZE                         {464, 16}
#define GDMA_ID_DST_HSIZE                         {480, 16}
#define GDMA_ID_DST_WSIZE                         {496, 16}
#define GDMA_ID_SRC_START_ADDR_L32                {512, 32}
#define GDMA_ID_SRC_START_ADDR_H8                 {544, 13}
#define GDMA_ID_DST_START_ADDR_L32                {576, 32}
#define GDMA_ID_DST_START_ADDR_H8                 {608, 13}
#define GDMA_ARE_DTYPE                            {640, 4}
#define GDMA_ARE_OPCODE                           {644, 4}
#define GDMA_ARE_PSUM_OP                          {648, 4}
#define GDMA_ID_MASK_START_ADDR_L32               {640, 32}  // Also for index
#define GDMA_ID_MASK_START_ADDR_H8                {672, 32}  // Also for index
#define GDMA_ID_LOCALMEM_MASK_L32                 {704, 32}
#define GDMA_ID_LOCALMEM_MASK_H32                 {736, 32}

// gdma control reg defines
#define GDMA_ID_CFG_DES_ENABLE                    {0, 1}
#define GDMA_ID_CFG_CMD_ID_RESET                  {1, 1}
#define GDMA_ID_CFG_DES_CLR                       {4, 1}
#define GDMA_ID_CFG_IST_FIFO_DEPTH                {8, 7}
#define GDMA_ID_CFG_BASE_MSGID                    {16, 9}
#define GDMA_ID_CFG_DES_MODE_END                  {128, 1}
#define GDMA_ID_CFG_CURRENT_CMD_ID                {288, 32}
#define GDMA_ID_CFG_FILTER_NUM                    {320, 32}
#define GDMA_ID_CFG_MST_SINGLESTEP_DISABLE        {392, 1}
#define GDMA_ID_CFG_SLV_SINGLESTEP_DISABLE        {393, 1}
#define GDMA_ID_CFG_MST_BREAKPOINT_DISABLE        {394, 1}
#define GDMA_ID_CFG_SLV_BREAKPOINT_DISABLE        {395, 1}
#define GDMA_ID_CFG_BASE_DDR0                     {416, 32}
#define GDMA_ID_CFG_BASE_DDR31                    {1408, 32}
#define GDMA_ID_CFG_SEED_L32                      {1952, 32}
#define GDMA_ID_CFG_SEED_H32                      {1984, 32}
#define GDMA_ID_CFG_SEED_EN                       {2016, 1}
#define GDMA_ID_CFG_MST_IRQ_SINGLESTEP            {2375, 1}
#define GDMA_ID_CFG_SLV_IRQ_SINGLESTEP            {2376, 1}
#define GDMA_ID_CFG_MST_IRQ_BREAKPOINT            {2377, 1}
#define GDMA_ID_CFG_SLV_IRQ_BREAKPOINT            {2378, 1}
#define GDMA_ID_CFG_MST_DBG_MODE                  {2560, 2}
#define GDMA_ID_CFG_SLV_DBG_MODE                  {2562, 2}
#define GDMA_ID_CFG_MST_DBG_EXEC_ENABLE           {2564, 1}
#define GDMA_ID_CFG_SLV_DBG_EXEC_ENABLE           {2565, 1}

// gdma pack defines
#define GDMA_PACK_INTR_EN(val)                    {GDMA_ID_INTR_EN, (val)}
#define GDMA_PACK_STRIDE_ENABLE(val)              {GDMA_ID_STRIDE_ENABLE, (val)}
#define GDMA_PACK_NCHW_COPY(val)                  {GDMA_ID_NCHW_COPY, (val)}
#define GDMA_PACK_CMD_SHORT(val)                  {GDMA_ID_CMD_SHORT, (val)}
#define GDMA_PACK_CMD_TYPE(val)                   {GDMA_ID_CMD_TYPE, (val)}
#define GDMA_PACK_CMD_SPECIAL_FUNCTION(val)       {GDMA_ID_CMD_SPECIAL_FUNCTION, (val)}
#define GDMA_PACK_FILL_CONSTANT_EN(val)           {GDMA_ID_FILL_CONSTANT_EN, (val)}
#define GDMA_PACK_SRC_DATA_FORMAT(val)            {GDMA_ID_SRC_DATA_FORMAT, (val)}
#define GDMA_PACK_MASK_DATA_FORMAT(val)           {GDMA_ID_MASK_DATA_FORMAT, (val)}
#define GDMA_PACK_RAND_MASK_INTER_STATE(val)      {GDMA_ID_RAND_MASK_INTER_STATE, (val)}
#define GDMA_PACK_RAND_MASK_INIT_SEED(val)        {GDMA_ID_RAND_MASK_INIT_SEED, (val)}
#define GDMA_PACK_CMD_ID_DEP(val)                 {GDMA_ID_CMD_ID_DEP, (val)}
#define GDMA_PACK_CONSTANT_VALUE(val)             {GDMA_ID_CONSTANT_VALUE, (val)}
#define GDMA_PACK_SRC_NSTRIDE(val)                {GDMA_ID_SRC_NSTRIDE, (val)}
#define GDMA_PACK_SRC_CSTRIDE(val)                {GDMA_ID_SRC_CSTRIDE, (val)}
#define GDMA_PACK_SRC_HSTRIDE(val)                {GDMA_ID_SRC_HSTRIDE, (val)}
#define GDMA_PACK_SRC_WSTRIDE(val)                {GDMA_ID_SRC_WSTRIDE, (val)}
#define GDMA_PACK_DST_NSTRIDE(val)                {GDMA_ID_DST_NSTRIDE, (val)}
#define GDMA_PACK_DST_CSTRIDE(val)                {GDMA_ID_DST_CSTRIDE, (val)}
#define GDMA_PACK_DST_HSTRIDE(val)                {GDMA_ID_DST_HSTRIDE, (val)}
#define GDMA_PACK_DST_WSTRIDE(val)                {GDMA_ID_DST_WSTRIDE, (val)}
#define GDMA_PACK_SRC_NSIZE(val)                  {GDMA_ID_SRC_NSIZE, (val)}
#define GDMA_PACK_SRC_CSIZE(val)                  {GDMA_ID_SRC_CSIZE, (val)}
#define GDMA_PACK_SRC_HSIZE(val)                  {GDMA_ID_SRC_HSIZE, (val)}
#define GDMA_PACK_SRC_WSIZE(val)                  {GDMA_ID_SRC_WSIZE, (val)}
#define GDMA_PACK_DST_NSIZE(val)                  {GDMA_ID_DST_NSIZE, (val)}
#define GDMA_PACK_DST_CSIZE(val)                  {GDMA_ID_DST_CSIZE, (val)}
#define GDMA_PACK_DST_HSIZE(val)                  {GDMA_ID_DST_HSIZE, (val)}
#define GDMA_PACK_DST_WSIZE(val)                  {GDMA_ID_DST_WSIZE, (val)}
#define GDMA_PACK_SRC_START_ADDR_L32(val)         {GDMA_ID_SRC_START_ADDR_L32, (val)}
#define GDMA_PACK_SRC_START_ADDR_H8(val)          {GDMA_ID_SRC_START_ADDR_H8, (val)}
#define GDMA_PACK_DST_START_ADDR_L32(val)         {GDMA_ID_DST_START_ADDR_L32, (val)}
#define GDMA_PACK_DST_START_ADDR_H8(val)          {GDMA_ID_DST_START_ADDR_H8, (val)}
#define GDMA_PACK_MASK_START_ADDR_L32(val)        {GDMA_ID_MASK_START_ADDR_L32, (val)}
#define GDMA_PACK_MASK_START_ADDR_H8(val)         {GDMA_ID_MASK_START_ADDR_H8, (val)}
#define GDMA_PACK_LOCALMEM_MASK_L32(val)          {GDMA_ID_LOCALMEM_MASK_L32, (val)}
#define GDMA_PACK_LOCALMEM_MASK_H32(val)          {GDMA_ID_LOCALMEM_MASK_H32, (val)}

#define GDMA_PACK_ARE_PSUM_OP(val)                {GDMA_ARE_PSUM_OP, (val)}
#define GDMA_PACK_ARE_OPCODE(val)                 {GDMA_ARE_OPCODE, (val)}

#endif  // __GDMA_REG_DEF_H__
