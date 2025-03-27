#ifndef __SORT_REG_DEF_H__
#define __SORT_REG_DEF_H__

typedef enum {
  HAU_NONE_MSG = 0,
  HAU_SEND_MSG = 1,
  HAU_WAIT_MSG = 2,
} HAU_MSG_TYPE;

// sort reg id defines
#define SORT_ID_BASE_ADDR_ID0                     {0, 32}
#define SORT_ID_PIO_ENABLE                        {1024, 1}
#define SORT_ID_INT_ENABLE                        {1025, 1}
#define SORT_ID_RESERVED0                         {1026, 30}
#define SORT_ID_DSCRP_START                       {1056, 1}
#define SORT_ID_DES_BASE_ADDR_RD_ENABLE           {1057, 1}
#define SORT_ID_DSCRP_START_ADDR_31_2             {1058, 30}
#define SORT_ID_DSCRP_START_ADDR_63_32            {1088, 32}
#define SORT_ID_OP_SYNC_ID_FINISH                 {1120, 32}
#define SORT_ID_PIO_BUF_DEPTH                     {1152, 4}
#define SORT_ID_RESERVED3                         {1156, 4}
#define SORT_ID_ENGINE_BUSY                       {1160, 1}
#define SORT_ID_RESERVED2                         {1161, 23}
#define SORT_ID_DSCRP_INT_STS                     {1184, 1}
#define SORT_ID_RESERVED4                         {1185, 31}
#define SORT_ID_REG_SYNC_ID_SW_CLR                {1216, 1}
#define SORT_ID_RESERVED5                         {1217, 31}
#define SORT_ID_BASE_MSGID                        {1248, 9}
#define SORT_ID_RESERVED6                         {1257, 23}
#define SORT_ID_DES_BASE_ADDR_31_0                {1280, 32}
#define SORT_ID_DES_BASE_ADDR_39_32               {1312, 8}


#define SORT_ID_DSCP_VLD                          {0, 1}
#define SORT_ID_RESERVED10                        {1, 7}
#define SORT_ID_ENABLE_SYNCID                     {8, 1}
#define SORT_ID_RESERVED9                         {9, 7}
#define SORT_ID_INT_STS_EN                        {16, 1}
#define SORT_ID_RESERVED8                         {17, 7}
#define SORT_ID_EOD                               {24, 1}
#define SORT_ID_RESERVED7                         {25, 7}
#define SORT_SRC_DATA_DDR_TAG                     {32, 8}
#define SORT_DST_DATA_DDR_TAG                     {40, 8}
#define SORT_SRC_INDEX_DDR_TAG                    {48, 8}
#define SORT_DST_INDEX_DDR_TAG                    {56, 8}
#define SORT_CNT_DDR_TAG                          {64, 8}
#define SORT_DATA_ROW_NUM                         {96, 32}
#define SORT_MSG_ACTION                           {320, 4}
#define SORT_WCNT_OR_SCNT                         {328, 7}
#define SORT_MSG_ID                               {335, 9}
#define SORT_ID_SORTCNT_ADDR_31_0                 {352, 32}
#define SORT_ID_CMP_DATA_FMT                      {384, 2}
#define SORT_2D_ENABLE                            {387, 1}
#define SORT_ID_UNIQUE_EN                         {388, 1}
#define SORT_ID_SORT_EN                           {389, 1}
#define SORT_ID_RESERVED15                        {390, 1}
#define SORT_ID_SORT_DESCEND_EN                   {391, 1}
#define SORT_ID_SORT_INDX_EN                      {392, 1}
#define SORT_ID_SORT_AUTO_INDX                    {393, 1}
#define SORT_ID_RESERVED14                        {394, 6}
#define SORT_ID_DEL_MIN_SORTCNT_EN                {400, 1}
#define SORT_ID_SORT_KTH_CMD                      {401, 1}
#define SORT_ID_RESERVED13                        {402, 1}
#define SORT_ID_RESERVED12                        {403, 1}
#define SORT_ID_SORT_RES_INDX_SEL                 {404, 1}
#define SORT_ID_DIS_WR_DST_SORT_IDX               {405, 1}
#define SORT_ID_DIS_WR_DST_SORT_DAT               {406, 1}
#define SORT_ID_DIS_WR_DST_SORTCNT                {407, 1}
#define SORT_ID_SORTCNT_ADDR_39_32                {408, 8}
#define SORT_ID_SRC_DATA_ADDR                     {416, 32}
#define SORT_ID_SRC_DATA_ADDR_39_32               {448, 8}
#define SORT_ID_DST_DATA_ADDR_39_32               {456, 8}
#define SORT_ID_SRC_INDEX_ADDR_39_32              {464, 8}
#define SORT_ID_DST_INDEX_ADDR_39_32              {472, 8}
#define SORT_ID_DST_DATA_ADDR_31_0                {480, 32}
#define SORT_ID_VALUE_TOP_M                       {512, 32}
#define SORT_ID_SRC_SIZE_DW                       {544, 32}
#define SORT_ID_SRC_INDEX_ADDR_31_0               {576, 32}
#define SORT_ID_DST_INDEX_ADDR_31_0               {608, 32}

// sort pack defines
#define SORT_PACK_RESERVED0(val)                  {SORT_ID_RESERVED, (val)}
#define SORT_PACK_INT_ENABLE(val)                 {SORT_ID_INT_ENABLE, (val)}
#define SORT_PACK_PIO_ENABLE(val)                 {SORT_ID_PIO_ENABLE, (val)}
#define SORT_PACK_DSCRP_START_ADDR_31_4(val)      {SORT_ID_DSCRP_START_ADDR_31_4, (val)}
#define SORT_PACK_RESERVED1(val)                  {SORT_ID_RESERVED, (val)}
#define SORT_PACK_DSCRP_START(val)                {SORT_ID_DSCRP_START, (val)}
#define SORT_PACK_DSCRP_START_ADDR_63_32(val)     {SORT_ID_DSCRP_START_ADDR_63_32, (val)}
#define SORT_PACK_OP_SYNC_ID_FINISH(val)          {SORT_ID_OP_SYNC_ID_FINISH, (val)}
#define SORT_PACK_RESERVED2(val)                  {SORT_ID_RESERVED, (val)}
#define SORT_PACK_ENGINE_BUSY(val)                {SORT_ID_ENGINE_BUSY, (val)}
#define SORT_PACK_RESERVED3(val)                  {SORT_ID_RESERVED, (val)}
#define SORT_PACK_PIO_BUF_DEPTH(val)              {SORT_ID_PIO_BUF_DEPTH, (val)}
#define SORT_PACK_RESERVED4(val)                  {SORT_ID_RESERVED, (val)}
#define SORT_PACK_DSCRP_INT_STS(val)              {SORT_ID_DSCRP_INT_STS, (val)}
#define SORT_PACK_RESERVED5(val)                  {SORT_ID_RESERVED, (val)}
#define SORT_PACK_REG_SYNC_ID_SW_CLR(val)         {SORT_ID_REG_SYNC_ID_SW_CLR, (val)}
#define SORT_PACK_RESERVED6(val)                  {SORT_ID_RESERVED, (val)}
#define SORT_PACK_REG_SYNC_ID_31_0(val)           {SORT_ID_REG_SYNC_ID_31_0, (val)}
#define SORT_PACK_REG_SYNC_ID_63_32(val)          {SORT_ID_REG_SYNC_ID_63_32, (val)}
#define SORT_PACK_REG_SYNC_ID_95_64(val)          {SORT_ID_REG_SYNC_ID_95_64, (val)}
#define SORT_PACK_REG_SYNC_ID_127_96(val)         {SORT_ID_REG_SYNC_ID_127_96, (val)}
#define SORT_PACK_REG_SYNC_ID_159_128(val)        {SORT_ID_REG_SYNC_ID_159_128, (val)}
#define SORT_PACK_REG_SYNC_ID_191_160(val)        {SORT_ID_REG_SYNC_ID_191_160, (val)}
#define SORT_PACK_REG_SYNC_ID_223_192(val)        {SORT_ID_REG_SYNC_ID_223_192, (val)}
#define SORT_PACK_REG_SYNC_ID_255_224(val)        {SORT_ID_REG_SYNC_ID_255_224, (val)}
#define SORT_PACK_RESERVED7(val)                  {SORT_ID_RESERVED, (val)}
#define SORT_PACK_EOD(val)                        {SORT_ID_EOD, (val)}
#define SORT_PACK_RESERVED8(val)                  {SORT_ID_RESERVED, (val)}
#define SORT_PACK_INT_STS_EN(val)                 {SORT_ID_INT_STS_EN, (val)}
#define SORT_PACK_RESERVED9(val)                  {SORT_ID_RESERVED, (val)}
#define SORT_PACK_ENABLE_SYNCID(val)              {SORT_ID_ENABLE_SYNCID, (val)}
#define SORT_PACK_RESERVED10(val)                 {SORT_ID_RESERVED, (val)}
#define SORT_PACK_DSCP_VLD(val)                   {SORT_ID_DSCP_VLD, (val)}
#define SORT_PACK_OP_SYNC_ID(val)                 {SORT_ID_OP_SYNC_ID, (val)}
#define SORT_PACK_EXP_SYNC_ID_0(val)              {SORT_ID_EXP_SYNC_ID_0, (val)}
#define SORT_PACK_EXP_SYNC_ID_1(val)              {SORT_ID_EXP_SYNC_ID_1, (val)}
#define SORT_PACK_EXP_SYNC_ID_2(val)              {SORT_ID_EXP_SYNC_ID_2, (val)}
#define SORT_PACK_EXP_SYNC_ID_3(val)              {SORT_ID_EXP_SYNC_ID_3, (val)}
#define SORT_PACK_EXP_SYNC_ID_4(val)              {SORT_ID_EXP_SYNC_ID_4, (val)}
#define SORT_PACK_EXP_SYNC_ID_5(val)              {SORT_ID_EXP_SYNC_ID_5, (val)}
#define SORT_PACK_EXP_SYNC_ID_6(val)              {SORT_ID_EXP_SYNC_ID_6, (val)}
#define SORT_PACK_EXP_SYNC_ID_7(val)              {SORT_ID_EXP_SYNC_ID_7, (val)}
#define SORT_PACK_SRC_OFFSET_INDX(val)            {SORT_ID_SRC_OFFSET_INDX, (val)}
#define SORT_PACK_SORTCNT_ADDR_31_0(val)          {SORT_ID_SORTCNT_ADDR_31_0, (val)}
#define SORT_PACK_SORTCNT_ADDR_39_32(val)         {SORT_ID_SORTCNT_ADDR_39_32, (val)}
#define SORT_PACK_DIS_WR_DST_SORTCNT(val)         {SORT_ID_DIS_WR_DST_SORTCNT, (val)}
#define SORT_PACK_DIS_WR_DST_SORT_DAT(val)        {SORT_ID_DIS_WR_DST_SORT_DAT, (val)}
#define SORT_PACK_DIS_WR_DST_SORT_IDX(val)        {SORT_ID_DIS_WR_DST_SORT_IDX, (val)}
#define SORT_PACK_SORT_RES_INDX_SEL(val)          {SORT_ID_SORT_RES_INDX_SEL, (val)}
#define SORT_PACK_RESERVED12(val)                 {SORT_ID_RESERVED, (val)}
#define SORT_PACK_RESERVED13(val)                 {SORT_ID_RESERVED, (val)}
#define SORT_PACK_SORT_KTH_CMD(val)               {SORT_ID_SORT_KTH_CMD, (val)}
#define SORT_PACK_DEL_MIN_SORTCNT_EN(val)         {SORT_ID_DEL_MIN_SORTCNT_EN, (val)}
#define SORT_PACK_DES_DLY_NUM_SW(val)             {SORT_ID_DES_DLY_NUM_SW, (val)}
#define SORT_PACK_DES_DLY_NUM_AUTO(val)           {SORT_ID_DES_DLY_NUM_AUTO, (val)}
#define SORT_PACK_RESERVED14(val)                 {SORT_ID_RESERVED, (val)}
#define SORT_PACK_SORT_AUTO_INDX(val)             {SORT_ID_SORT_AUTO_INDX, (val)}
#define SORT_PACK_SORT_INDX_EN(val)               {SORT_ID_SORT_INDX_EN, (val)}
#define SORT_PACK_SORT_DESCEND_EN(val)            {SORT_ID_SORT_DESCEND_EN, (val)}
#define SORT_PACK_RESERVED15(val)                 {SORT_ID_RESERVED, (val)}
#define SORT_PACK_SORT_EN(val)                    {SORT_ID_SORT_EN, (val)}
#define SORT_PACK_UNIQUE_EN(val)                  {SORT_ID_UNIQUE_EN, (val)}
#define SORT_PACK_CMP_CHAIN_NUM(val)              {SORT_ID_CMP_CHAIN_NUM, (val)}
#define SORT_PACK_CMP_DATA_FMT(val)               {SORT_ID_CMP_DATA_FMT, (val)}
#define SORT_PACK_SRC_DATA_ADDR(val)              {SORT_ID_SRC_DATA_ADDR, (val)}
#define SORT_PACK_DST_INDEX_ADDR_39_32(val)       {SORT_ID_DST_INDEX_ADDR_39_32, (val)}
#define SORT_PACK_SRC_INDEX_ADDR_39_32(val)       {SORT_ID_SRC_INDEX_ADDR_39_32, (val)}
#define SORT_PACK_DST_DATA_ADDR_39_32(val)        {SORT_ID_DST_DATA_ADDR_39_32, (val)}
#define SORT_PACK_SRC_DATA_ADDR_39_32(val)        {SORT_ID_SRC_DATA_ADDR_39_32, (val)}
#define SORT_PACK_DST_DATA_ADDR_31_0(val)         {SORT_ID_DST_DATA_ADDR_31_0, (val)}
#define SORT_PACK_VALUE_TOP_M(val)                {SORT_ID_VALUE_TOP_M, (val)}
#define SORT_PACK_SRC_SIZE_DW(val)                {SORT_ID_SRC_SIZE_DW, (val)}
#define SORT_PACK_SRC_INDEX_ADDR_31_0(val)        {SORT_ID_SRC_INDEX_ADDR_31_0, (val)}
#define SORT_PACK_DST_INDEX_ADDR_31_0(val)        {SORT_ID_DST_INDEX_ADDR_31_0, (val)}

#endif  // __SORT_REG_DEF_H__
